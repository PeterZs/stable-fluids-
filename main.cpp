#include "src/stable_fluids.h"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << what << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

bool stable_ok(int32_t code, const char* what, const StableFluidsContext* context = nullptr) {
    if (code == STABLE_FLUIDS_SUCCESS) {
        return true;
    }

    const uint64_t message_length =
        context != nullptr ? stable_fluids_context_last_error_length(context) : stable_fluids_last_error_length();
    std::vector<char> message(static_cast<std::size_t>(message_length + 1), '\0');

    const int32_t copy_code = context != nullptr
        ? stable_fluids_copy_context_last_error(context, message.data(), static_cast<uint64_t>(message.size()))
        : stable_fluids_copy_last_error(message.data(), static_cast<uint64_t>(message.size()));

    std::cerr << what << " failed (" << code << ")";
    if (copy_code == STABLE_FLUIDS_SUCCESS && !message.empty() && message[0] != '\0') {
        std::cerr << ": " << message.data();
    }
    std::cerr << '\n';
    return false;
}

StableFluidsBufferView make_f32_device_buffer(void* data, uint64_t size_bytes) {
    StableFluidsBufferView view{};
    view.data = data;
    view.size_bytes = size_bytes;
    view.format = STABLE_FLUIDS_BUFFER_FORMAT_F32;
    view.memory_type = STABLE_FLUIDS_MEMORY_TYPE_CUDA_DEVICE;
    return view;
}

}  // namespace

int main() {
    StableFluidsContextDesc context_desc = stable_fluids_context_desc_default();
    context_desc.nx = 96;
    context_desc.ny = 96;
    context_desc.nz = 96;
    context_desc.dt = 1.0f / 60.0f;
    context_desc.cell_size = 1.0f;
    context_desc.viscosity = 0.00015f;
    context_desc.diffusion = 0.00005f;
    context_desc.diffuse_iterations = 24;
    context_desc.pressure_iterations = 96;
    context_desc.pressure_tolerance = 1.0e-5f;

    StableFluidsContext* context = stable_fluids_context_create(&context_desc);
    if (context == nullptr) {
        stable_ok(STABLE_FLUIDS_ERROR_RUNTIME, "stable_fluids_context_create");
        return EXIT_FAILURE;
    }

    const uint64_t element_count = stable_fluids_context_required_elements(context);
    const uint64_t field_bytes = stable_fluids_context_required_scalar_field_bytes(context);

    float* density = nullptr;
    float* velocity_x = nullptr;
    float* velocity_y = nullptr;
    float* velocity_z = nullptr;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&density), field_bytes), "cudaMalloc density") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_x), field_bytes), "cudaMalloc velocity_x") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_y), field_bytes), "cudaMalloc velocity_y") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_z), field_bytes), "cudaMalloc velocity_z")) {
        cudaFree(density);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        stable_fluids_context_destroy(context);
        return EXIT_FAILURE;
    }

    StableFluidsFieldSetDesc fields{};
    fields.density = make_f32_device_buffer(density, field_bytes);
    fields.velocity_x = make_f32_device_buffer(velocity_x, field_bytes);
    fields.velocity_y = make_f32_device_buffer(velocity_y, field_bytes);
    fields.velocity_z = make_f32_device_buffer(velocity_z, field_bytes);

    StableFluidsDensitySplatDesc density_splat{};
    density_splat.center_x = static_cast<float>(context_desc.nx) * 0.5f;
    density_splat.center_y = static_cast<float>(context_desc.ny) * 0.33f;
    density_splat.center_z = static_cast<float>(context_desc.nz) * 0.5f;
    density_splat.radius = 5.0f;
    density_splat.amount = 6.0f;

    StableFluidsForceSplatDesc force_splat{};
    force_splat.center_x = density_splat.center_x;
    force_splat.center_y = density_splat.center_y;
    force_splat.center_z = density_splat.center_z;
    force_splat.radius = 6.0f;
    force_splat.force_x = 1.25f;
    force_splat.force_y = 2.5f;
    force_splat.force_z = 0.75f;

    if (!stable_ok(stable_fluids_fields_clear(context, &fields), "stable_fluids_fields_clear", context) ||
        !stable_ok(stable_fluids_fields_add_density_splat(context, &fields, &density_splat), "stable_fluids_fields_add_density_splat", context) ||
        !stable_ok(stable_fluids_fields_add_force_splat(context, &fields, &force_splat), "stable_fluids_fields_add_force_splat", context)) {
        cudaFree(density);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        stable_fluids_context_destroy(context);
        return EXIT_FAILURE;
    }

    for (int frame = 0; frame < 24; ++frame) {
        if (frame < 8) {
            StableFluidsDensitySplatDesc pulse_density = density_splat;
            pulse_density.radius = 4.0f;
            pulse_density.amount = 1.5f;

            StableFluidsForceSplatDesc pulse_force = force_splat;
            pulse_force.radius = 4.0f;
            pulse_force.force_x = 0.0f;
            pulse_force.force_y = 0.5f;
            pulse_force.force_z = 0.0f;

            if (!stable_ok(stable_fluids_fields_add_density_splat(context, &fields, &pulse_density), "stable_fluids_fields_add_density_splat", context) ||
                !stable_ok(stable_fluids_fields_add_force_splat(context, &fields, &pulse_force), "stable_fluids_fields_add_force_splat", context)) {
                cudaFree(density);
                cudaFree(velocity_x);
                cudaFree(velocity_y);
                cudaFree(velocity_z);
                stable_fluids_context_destroy(context);
                return EXIT_FAILURE;
            }
        }

        if (!stable_ok(stable_fluids_fields_step(context, &fields), "stable_fluids_fields_step", context)) {
            cudaFree(density);
            cudaFree(velocity_x);
            cudaFree(velocity_y);
            cudaFree(velocity_z);
            stable_fluids_context_destroy(context);
            return EXIT_FAILURE;
        }
    }

    std::vector<float> host_density(static_cast<std::size_t>(element_count), 0.0f);
    if (!cuda_ok(cudaMemcpy(host_density.data(), density, field_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy density to host")) {
        cudaFree(density);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        stable_fluids_context_destroy(context);
        return EXIT_FAILURE;
    }

    const float total_density = std::accumulate(host_density.begin(), host_density.end(), 0.0f);
    const float peak_density = host_density.empty() ? 0.0f : *std::max_element(host_density.begin(), host_density.end());

    std::cout << "stable-fluids-app\n";
    std::cout << "grid: " << context_desc.nx << " x " << context_desc.ny << " x " << context_desc.nz << '\n';
    std::cout << "elements: " << static_cast<unsigned long long>(element_count) << '\n';
    std::cout << "total density: " << total_density << '\n';
    std::cout << "peak density: " << peak_density << '\n';

    cudaFree(density);
    cudaFree(velocity_x);
    cudaFree(velocity_y);
    cudaFree(velocity_z);
    stable_fluids_context_destroy(context);
    return EXIT_SUCCESS;
}
