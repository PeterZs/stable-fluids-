#include "stable-fluids.h"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

#include <nvtx3/nvtx3.hpp>

int32_t stable_demo_add_source_async(void* density, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, float center_x, float center_y, float center_z, float radius, float density_amount, float velocity_source_x, float velocity_source_y, float velocity_source_z, int32_t block_x, int32_t block_y, int32_t block_z,
    void* cuda_stream);

int main() {
    nvtx3::scoped_range app_range{"stable.demo"};
    auto cuda_ok = [](const cudaError_t status, const char* what) {
        if (status == cudaSuccess) return true;
        std::cerr << what << " failed: " << cudaGetErrorString(status) << '\n';
        return false;
    };
    auto stable_ok = [](const int32_t code, const char* what) {
        if (code == 0) return true;
        std::cerr << what << " failed (" << code << ")\n";
        return false;
    };

    constexpr int32_t nx                  = 96;
    constexpr int32_t ny                  = 96;
    constexpr int32_t nz                  = 96;
    constexpr float cell_size             = 1.0f;
    constexpr float dt                    = 1.0f / 60.0f;
    constexpr float viscosity             = 0.00015f;
    constexpr float diffusion             = 0.00005f;
    constexpr int32_t diffuse_iterations  = 24;
    constexpr int32_t pressure_iterations = 96;
    constexpr int32_t block_x             = 8;
    constexpr int32_t block_y             = 8;
    constexpr int32_t block_z             = 8;

    const uint64_t scalar_bytes           = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_x_bytes       = static_cast<uint64_t>(nx + 1) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_y_bytes       = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny + 1) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_z_bytes       = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz + 1) * sizeof(float);
    const uint64_t element_count          = scalar_bytes / sizeof(float);

    float* density                       = nullptr;
    float* velocity_x                    = nullptr;
    float* velocity_y                    = nullptr;
    float* velocity_z                    = nullptr;
    float* temporary_density             = nullptr;
    float* temporary_velocity_x          = nullptr;
    float* temporary_velocity_y          = nullptr;
    float* temporary_velocity_z          = nullptr;
    float* temporary_previous_density    = nullptr;
    float* temporary_previous_velocity_x = nullptr;
    float* temporary_previous_velocity_y = nullptr;
    float* temporary_previous_velocity_z = nullptr;
    float* temporary_pressure            = nullptr;
    float* temporary_divergence          = nullptr;
    cudaStream_t stream                  = nullptr;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&density), scalar_bytes), "cudaMalloc density") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_x), velocity_x_bytes), "cudaMalloc velocity_x") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_y), velocity_y_bytes), "cudaMalloc velocity_y")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_z), velocity_z_bytes), "cudaMalloc velocity_z") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_density), scalar_bytes), "cudaMalloc temporary_density")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_velocity_x), velocity_x_bytes), "cudaMalloc temporary_velocity_x") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_velocity_y), velocity_y_bytes), "cudaMalloc temporary_velocity_y")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_velocity_z), velocity_z_bytes), "cudaMalloc temporary_velocity_z") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_density), scalar_bytes), "cudaMalloc temporary_previous_density")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_x), velocity_x_bytes), "cudaMalloc temporary_previous_velocity_x") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_y), velocity_y_bytes), "cudaMalloc temporary_previous_velocity_y")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_z), velocity_z_bytes), "cudaMalloc temporary_previous_velocity_z") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_pressure), scalar_bytes), "cudaMalloc temporary_pressure")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_divergence), scalar_bytes), "cudaMalloc temporary_divergence") || !cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) {
        cudaFree(density);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        cudaFree(temporary_density);
        cudaFree(temporary_velocity_x);
        cudaFree(temporary_velocity_y);
        cudaFree(temporary_velocity_z);
        cudaFree(temporary_previous_density);
        cudaFree(temporary_previous_velocity_x);
        cudaFree(temporary_previous_velocity_y);
        cudaFree(temporary_previous_velocity_z);
        cudaFree(temporary_pressure);
        cudaFree(temporary_divergence);
        if (stream != nullptr) cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    if (!cuda_ok(cudaMemsetAsync(density, 0, scalar_bytes, stream), "cudaMemsetAsync density") || !cuda_ok(cudaMemsetAsync(velocity_x, 0, velocity_x_bytes, stream), "cudaMemsetAsync velocity_x") || !cuda_ok(cudaMemsetAsync(velocity_y, 0, velocity_y_bytes, stream), "cudaMemsetAsync velocity_y")
        || !cuda_ok(cudaMemsetAsync(velocity_z, 0, velocity_z_bytes, stream), "cudaMemsetAsync velocity_z")
        || !stable_ok(stable_demo_add_source_async(density, velocity_x, velocity_y, velocity_z, nx, ny, nz, static_cast<float>(nx) * 0.5f, static_cast<float>(ny) * 0.33f, static_cast<float>(nz) * 0.5f, 5.0f, 6.0f, 1.25f, 2.5f, 0.75f, block_x, block_y, block_z, stream), "stable_demo_add_source_async")) {
        cudaStreamDestroy(stream);
        cudaFree(density);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        cudaFree(temporary_density);
        cudaFree(temporary_velocity_x);
        cudaFree(temporary_velocity_y);
        cudaFree(temporary_velocity_z);
        cudaFree(temporary_previous_density);
        cudaFree(temporary_previous_velocity_x);
        cudaFree(temporary_previous_velocity_y);
        cudaFree(temporary_previous_velocity_z);
        cudaFree(temporary_pressure);
        cudaFree(temporary_divergence);
        return EXIT_FAILURE;
    }

    for (int frame = 0; frame < 24; ++frame) {
        nvtx3::scoped_range frame_range{"stable.demo.frame"};
        if (frame < 8) {
            if (!stable_ok(stable_demo_add_source_async(density, velocity_x, velocity_y, velocity_z, nx, ny, nz, static_cast<float>(nx) * 0.5f, static_cast<float>(ny) * 0.33f, static_cast<float>(nz) * 0.5f, 4.0f, 1.5f, 0.0f, 0.5f, 0.0f, block_x, block_y, block_z, stream), "stable_demo_add_source_async")) {
                cudaStreamDestroy(stream);
                cudaFree(density);
                cudaFree(velocity_x);
                cudaFree(velocity_y);
                cudaFree(velocity_z);
                cudaFree(temporary_density);
                cudaFree(temporary_velocity_x);
                cudaFree(temporary_velocity_y);
                cudaFree(temporary_velocity_z);
                cudaFree(temporary_previous_density);
                cudaFree(temporary_previous_velocity_x);
                cudaFree(temporary_previous_velocity_y);
                cudaFree(temporary_previous_velocity_z);
                cudaFree(temporary_pressure);
                cudaFree(temporary_divergence);
                return EXIT_FAILURE;
            }
        }

        StableFluidsStepDesc step_desc{};
        step_desc.struct_size                  = sizeof(StableFluidsStepDesc);
        step_desc.api_version                  = 1;
        step_desc.nx                           = nx;
        step_desc.ny                           = ny;
        step_desc.nz                           = nz;
        step_desc.cell_size                    = cell_size;
        step_desc.dt                           = dt;
        step_desc.viscosity                    = viscosity;
        step_desc.diffusion                    = diffusion;
        step_desc.diffuse_iterations           = diffuse_iterations;
        step_desc.pressure_iterations          = pressure_iterations;
        step_desc.density                      = density;
        step_desc.velocity_x                   = velocity_x;
        step_desc.velocity_y                   = velocity_y;
        step_desc.velocity_z                   = velocity_z;
        step_desc.temporary_density            = temporary_density;
        step_desc.temporary_velocity_x         = temporary_velocity_x;
        step_desc.temporary_velocity_y         = temporary_velocity_y;
        step_desc.temporary_velocity_z         = temporary_velocity_z;
        step_desc.temporary_previous_density   = temporary_previous_density;
        step_desc.temporary_previous_velocity_x = temporary_previous_velocity_x;
        step_desc.temporary_previous_velocity_y = temporary_previous_velocity_y;
        step_desc.temporary_previous_velocity_z = temporary_previous_velocity_z;
        step_desc.temporary_pressure           = temporary_pressure;
        step_desc.temporary_divergence         = temporary_divergence;
        step_desc.block_x                      = block_x;
        step_desc.block_y                      = block_y;
        step_desc.block_z                      = block_z;
        step_desc.stream                       = stream;
        if (!stable_ok(stable_fluids_step_cuda(&step_desc), "stable_fluids_step_cuda")) {
            cudaStreamDestroy(stream);
            cudaFree(density);
            cudaFree(velocity_x);
            cudaFree(velocity_y);
            cudaFree(velocity_z);
            cudaFree(temporary_density);
            cudaFree(temporary_velocity_x);
            cudaFree(temporary_velocity_y);
            cudaFree(temporary_velocity_z);
            cudaFree(temporary_previous_density);
            cudaFree(temporary_previous_velocity_x);
            cudaFree(temporary_previous_velocity_y);
            cudaFree(temporary_previous_velocity_z);
            cudaFree(temporary_pressure);
            cudaFree(temporary_divergence);
            return EXIT_FAILURE;
        }
    }

    if (!cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
        cudaStreamDestroy(stream);
        cudaFree(density);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        cudaFree(temporary_density);
        cudaFree(temporary_velocity_x);
        cudaFree(temporary_velocity_y);
        cudaFree(temporary_velocity_z);
        cudaFree(temporary_previous_density);
        cudaFree(temporary_previous_velocity_x);
        cudaFree(temporary_previous_velocity_y);
        cudaFree(temporary_previous_velocity_z);
        cudaFree(temporary_pressure);
        cudaFree(temporary_divergence);
        return EXIT_FAILURE;
    }

    std::vector<float> host_density(static_cast<std::size_t>(element_count), 0.0f);
    if (!cuda_ok(cudaMemcpy(host_density.data(), density, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy density")) {
        cudaStreamDestroy(stream);
        cudaFree(density);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        cudaFree(temporary_density);
        cudaFree(temporary_velocity_x);
        cudaFree(temporary_velocity_y);
        cudaFree(temporary_velocity_z);
        cudaFree(temporary_previous_density);
        cudaFree(temporary_previous_velocity_x);
        cudaFree(temporary_previous_velocity_y);
        cudaFree(temporary_previous_velocity_z);
        cudaFree(temporary_pressure);
        cudaFree(temporary_divergence);
        return EXIT_FAILURE;
    }

    const float total_density = std::accumulate(host_density.begin(), host_density.end(), 0.0f);
    const float peak_density  = host_density.empty() ? 0.0f : *std::max_element(host_density.begin(), host_density.end());

    std::cout << "stable-fluids-app\n";
    std::cout << "grid: " << nx << " x " << ny << " x " << nz << '\n';
    std::cout << "elements: " << static_cast<unsigned long long>(element_count) << '\n';
    std::cout << "total density: " << total_density << '\n';
    std::cout << "peak density: " << peak_density << '\n';

    cudaStreamDestroy(stream);
    cudaFree(density);
    cudaFree(velocity_x);
    cudaFree(velocity_y);
    cudaFree(velocity_z);
    cudaFree(temporary_density);
    cudaFree(temporary_velocity_x);
    cudaFree(temporary_velocity_y);
    cudaFree(temporary_velocity_z);
    cudaFree(temporary_previous_density);
    cudaFree(temporary_previous_velocity_x);
    cudaFree(temporary_previous_velocity_y);
    cudaFree(temporary_previous_velocity_z);
    cudaFree(temporary_pressure);
    cudaFree(temporary_divergence);
    return EXIT_SUCCESS;
}
