#include "stable-fluids.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#include <nvtx3/nvtx3.hpp>

namespace stable_fluids {
    using Stream = cudaStream_t;

    namespace {

        int32_t cuda_code(const cudaError_t status) noexcept {
            return status == cudaSuccess ? 0 : 5001;
        }

        std::uint64_t scalar_bytes(const int32_t nx, const int32_t ny, const int32_t nz) {
            return static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
        }

        std::uint64_t velocity_x_bytes(const int32_t nx, const int32_t ny, const int32_t nz) {
            return static_cast<std::uint64_t>(nx + 1) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
        }

        std::uint64_t velocity_y_bytes(const int32_t nx, const int32_t ny, const int32_t nz) {
            return static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny + 1) * static_cast<std::uint64_t>(nz) * sizeof(float);
        }

        std::uint64_t velocity_z_bytes(const int32_t nx, const int32_t ny, const int32_t nz) {
            return static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz + 1) * sizeof(float);
        }

        dim3 make_grid(const int nx, const int ny, const int nz, const dim3& block) {
            return dim3(static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
        }

        __host__ __device__ int clampi(const int value, const int lo, const int hi) {
            return value < lo ? lo : (value > hi ? hi : value);
        }

        __host__ __device__ float clampf(const float value, const float lo, const float hi) {
            return value < lo ? lo : (value > hi ? hi : value);
        }

        __host__ __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
            return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
        }

        __device__ float fetch_clamped(const float* field, const int x, const int y, const int z, const int sx, const int sy, const int sz) {
            return field[index_3d(clampi(x, 0, sx - 1), clampi(y, 0, sy - 1), clampi(z, 0, sz - 1), sx, sy)];
        }

        __device__ float sample_grid(const float* field, float gx, float gy, float gz, const int sx, const int sy, const int sz) {
            gx = clampf(gx, 0.0f, static_cast<float>(sx - 1));
            gy = clampf(gy, 0.0f, static_cast<float>(sy - 1));
            gz = clampf(gz, 0.0f, static_cast<float>(sz - 1));

            const int x0 = clampi(static_cast<int>(floorf(gx)), 0, sx - 1);
            const int y0 = clampi(static_cast<int>(floorf(gy)), 0, sy - 1);
            const int z0 = clampi(static_cast<int>(floorf(gz)), 0, sz - 1);
            const int x1 = min(x0 + 1, sx - 1);
            const int y1 = min(y0 + 1, sy - 1);
            const int z1 = min(z0 + 1, sz - 1);
            const float tx = gx - static_cast<float>(x0);
            const float ty = gy - static_cast<float>(y0);
            const float tz = gz - static_cast<float>(z0);

            const float c000 = field[index_3d(x0, y0, z0, sx, sy)];
            const float c100 = field[index_3d(x1, y0, z0, sx, sy)];
            const float c010 = field[index_3d(x0, y1, z0, sx, sy)];
            const float c110 = field[index_3d(x1, y1, z0, sx, sy)];
            const float c001 = field[index_3d(x0, y0, z1, sx, sy)];
            const float c101 = field[index_3d(x1, y0, z1, sx, sy)];
            const float c011 = field[index_3d(x0, y1, z1, sx, sy)];
            const float c111 = field[index_3d(x1, y1, z1, sx, sy)];

            const float c00 = c000 + (c100 - c000) * tx;
            const float c10 = c010 + (c110 - c010) * tx;
            const float c01 = c001 + (c101 - c001) * tx;
            const float c11 = c011 + (c111 - c011) * tx;
            const float c0 = c00 + (c10 - c00) * ty;
            const float c1 = c01 + (c11 - c01) * ty;
            return c0 + (c1 - c0) * tz;
        }

        __device__ float sample_scalar(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h) {
            return sample_grid(field, pos.x / h - 0.5f, pos.y / h - 0.5f, pos.z / h - 0.5f, nx, ny, nz);
        }

        __device__ float sample_u(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h) {
            return sample_grid(field, pos.x / h, pos.y / h - 0.5f, pos.z / h - 0.5f, nx + 1, ny, nz);
        }

        __device__ float sample_v(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h) {
            return sample_grid(field, pos.x / h - 0.5f, pos.y / h, pos.z / h - 0.5f, nx, ny + 1, nz);
        }

        __device__ float sample_w(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h) {
            return sample_grid(field, pos.x / h - 0.5f, pos.y / h - 0.5f, pos.z / h, nx, ny, nz + 1);
        }

        __device__ float3 clamp_domain(const float3 pos, const int nx, const int ny, const int nz, const float h) {
            return make_float3(clampf(pos.x, 0.0f, static_cast<float>(nx) * h), clampf(pos.y, 0.0f, static_cast<float>(ny) * h), clampf(pos.z, 0.0f, static_cast<float>(nz) * h));
        }

        __device__ float3 sample_velocity(const float* velocity_x, const float* velocity_y, const float* velocity_z, float3 pos, const int nx, const int ny, const int nz, const float h) {
            pos = clamp_domain(pos, nx, ny, nz, h);
            return make_float3(sample_u(velocity_x, pos, nx, ny, nz, h), sample_v(velocity_y, pos, nx, ny, nz, h), sample_w(velocity_z, pos, nx, ny, nz, h));
        }

        __global__ void set_u_boundary_kernel(float* velocity_x, const int nx, const int ny, const int nz) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x <= nx && y < ny && z < nz && (x == 0 || x == nx)) velocity_x[index_3d(x, y, z, nx + 1, ny)] = 0.0f;
        }

        __global__ void set_v_boundary_kernel(float* velocity_y, const int nx, const int ny, const int nz) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x < nx && y <= ny && z < nz && (y == 0 || y == ny)) velocity_y[index_3d(x, y, z, nx, ny + 1)] = 0.0f;
        }

        __global__ void set_w_boundary_kernel(float* velocity_z, const int nx, const int ny, const int nz) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x < nx && y < ny && z <= nz && (z == 0 || z == nz)) velocity_z[index_3d(x, y, z, nx, ny)] = 0.0f;
        }

        __global__ void advect_u_kernel(float* destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x > nx || y >= ny || z >= nz) return;
            const float3 pos = make_float3(static_cast<float>(x) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
            const float3 velocity = sample_velocity(source_x, source_y, source_z, pos, nx, ny, nz, h);
            destination[index_3d(x, y, z, nx + 1, ny)] = sample_u(source_x, clamp_domain(make_float3(pos.x - dt * velocity.x, pos.y - dt * velocity.y, pos.z - dt * velocity.z), nx, ny, nz, h), nx, ny, nz, h);
        }

        __global__ void advect_v_kernel(float* destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y > ny || z >= nz) return;
            const float3 pos = make_float3((static_cast<float>(x) + 0.5f) * h, static_cast<float>(y) * h, (static_cast<float>(z) + 0.5f) * h);
            const float3 velocity = sample_velocity(source_x, source_y, source_z, pos, nx, ny, nz, h);
            destination[index_3d(x, y, z, nx, ny + 1)] = sample_v(source_y, clamp_domain(make_float3(pos.x - dt * velocity.x, pos.y - dt * velocity.y, pos.z - dt * velocity.z), nx, ny, nz, h), nx, ny, nz, h);
        }

        __global__ void advect_w_kernel(float* destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z > nz) return;
            const float3 pos = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, static_cast<float>(z) * h);
            const float3 velocity = sample_velocity(source_x, source_y, source_z, pos, nx, ny, nz, h);
            destination[index_3d(x, y, z, nx, ny)] = sample_w(source_z, clamp_domain(make_float3(pos.x - dt * velocity.x, pos.y - dt * velocity.y, pos.z - dt * velocity.z), nx, ny, nz, h), nx, ny, nz, h);
        }

        __global__ void advect_scalar_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const float dt) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;
            const float3 pos = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
            const float3 velocity = sample_velocity(velocity_x, velocity_y, velocity_z, pos, nx, ny, nz, h);
            destination[index_3d(x, y, z, nx, ny)] = fmaxf(0.0f, sample_scalar(source, clamp_domain(make_float3(pos.x - dt * velocity.x, pos.y - dt * velocity.y, pos.z - dt * velocity.z), nx, ny, nz, h), nx, ny, nz, h));
        }

        __global__ void diffuse_grid_kernel(float* destination, const float* source, const int sx, const int sy, const int sz, const float alpha, const float denom, const int parity) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= sx || y >= sy || z >= sz || ((x + y + z) & 1) != parity) return;
            const float neighbors = fetch_clamped(destination, x - 1, y, z, sx, sy, sz) + fetch_clamped(destination, x + 1, y, z, sx, sy, sz) + fetch_clamped(destination, x, y - 1, z, sx, sy, sz) + fetch_clamped(destination, x, y + 1, z, sx, sy, sz) + fetch_clamped(destination, x, y, z - 1, sx, sy, sz) + fetch_clamped(destination, x, y, z + 1, sx, sy, sz);
            destination[index_3d(x, y, z, sx, sy)] = (source[index_3d(x, y, z, sx, sy)] + alpha * neighbors) / denom;
        }

        __global__ void compute_divergence_kernel(float* divergence, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;
            divergence[index_3d(x, y, z, nx, ny)] = (fetch_clamped(velocity_x, x + 1, y, z, nx + 1, ny, nz) - fetch_clamped(velocity_x, x, y, z, nx + 1, ny, nz) + fetch_clamped(velocity_y, x, y + 1, z, nx, ny + 1, nz) - fetch_clamped(velocity_y, x, y, z, nx, ny + 1, nz)
                + fetch_clamped(velocity_z, x, y, z + 1, nx, ny, nz + 1) - fetch_clamped(velocity_z, x, y, z, nx, ny, nz + 1)) / h;
        }

        __global__ void pressure_rbgs_kernel(float* pressure, const float* divergence, const int nx, const int ny, const int nz, const float h, const int parity) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz || ((x + y + z) & 1) != parity) return;

            float sum = 0.0f;
            int count = 0;
            if (x > 0) {
                sum += pressure[index_3d(x - 1, y, z, nx, ny)];
                ++count;
            }
            if (x + 1 < nx) {
                sum += pressure[index_3d(x + 1, y, z, nx, ny)];
                ++count;
            }
            if (y > 0) {
                sum += pressure[index_3d(x, y - 1, z, nx, ny)];
                ++count;
            }
            if (y + 1 < ny) {
                sum += pressure[index_3d(x, y + 1, z, nx, ny)];
                ++count;
            }
            if (z > 0) {
                sum += pressure[index_3d(x, y, z - 1, nx, ny)];
                ++count;
            }
            if (z + 1 < nz) {
                sum += pressure[index_3d(x, y, z + 1, nx, ny)];
                ++count;
            }
            pressure[index_3d(x, y, z, nx, ny)] = count > 0 ? (sum - divergence[index_3d(x, y, z, nx, ny)] * h * h) / static_cast<float>(count) : 0.0f;
        }

        __global__ void subtract_gradient_u_kernel(float* velocity_x, const float* pressure, const int nx, const int ny, const int nz, const float h) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x > 0 && x < nx && y < ny && z < nz) velocity_x[index_3d(x, y, z, nx + 1, ny)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x - 1, y, z, nx, ny)]) / h;
        }

        __global__ void subtract_gradient_v_kernel(float* velocity_y, const float* pressure, const int nx, const int ny, const int nz, const float h) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x < nx && y > 0 && y < ny && z < nz) velocity_y[index_3d(x, y, z, nx, ny + 1)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y - 1, z, nx, ny)]) / h;
        }

        __global__ void subtract_gradient_w_kernel(float* velocity_z, const float* pressure, const int nx, const int ny, const int nz, const float h) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x < nx && y < ny && z > 0 && z < nz) velocity_z[index_3d(x, y, z, nx, ny)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y, z - 1, nx, ny)]) / h;
        }

    } // namespace

} // namespace stable_fluids

namespace {

    stable_fluids::Stream to_stream(void* cuda_stream) noexcept {
        return reinterpret_cast<stable_fluids::Stream>(cuda_stream);
    }

} // namespace

extern "C" {

int32_t stable_fluids_validate_desc(const StableFluidsStepDesc* desc) {
    if (desc == nullptr) return 1000;
    if (desc->struct_size < sizeof(StableFluidsStepDesc)) return 1000;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->cell_size <= 0.0f) return 1002;
    if (desc->dt <= 0.0f) return 1003;
    if (desc->diffuse_iterations <= 0 || desc->pressure_iterations <= 0) return 1004;
    if (desc->density == nullptr) return 2001;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    if (desc->temporary_density == nullptr) return 2007;
    if (desc->temporary_velocity_x == nullptr) return 2008;
    if (desc->temporary_velocity_y == nullptr) return 2009;
    if (desc->temporary_velocity_z == nullptr) return 2010;
    if (desc->temporary_previous_density == nullptr) return 2011;
    if (desc->temporary_previous_velocity_x == nullptr) return 2012;
    if (desc->temporary_previous_velocity_y == nullptr) return 2013;
    if (desc->temporary_previous_velocity_z == nullptr) return 2014;
    if (desc->temporary_pressure == nullptr) return 2015;
    if (desc->temporary_divergence == nullptr) return 2016;
    return 0;
}

int32_t stable_fluids_step_cuda(const StableFluidsStepDesc* desc) {
    using namespace stable_fluids;
    const int32_t nx = desc->nx;
    const int32_t ny = desc->ny;
    const int32_t nz = desc->nz;
    const float cell_size = desc->cell_size;
    const float dt = desc->dt;
    const float viscosity = desc->viscosity;
    const float diffusion = desc->diffusion;
    const int32_t diffuse_iterations = desc->diffuse_iterations;
    const int32_t pressure_iterations = desc->pressure_iterations;
    const int32_t block_x = desc->block_x;
    const int32_t block_y = desc->block_y;
    const int32_t block_z = desc->block_z;

    const auto cell_bytes = scalar_bytes(nx, ny, nz);
    const auto velocity_x_field_bytes = velocity_x_bytes(nx, ny, nz);
    const auto velocity_y_field_bytes = velocity_y_bytes(nx, ny, nz);
    const auto velocity_z_field_bytes = velocity_z_bytes(nx, ny, nz);

    auto* density_field = reinterpret_cast<float*>(desc->density);
    auto* density_temporary = reinterpret_cast<float*>(desc->temporary_density);
    auto* density_previous = reinterpret_cast<float*>(desc->temporary_previous_density);
    auto* velocity_x_field = reinterpret_cast<float*>(desc->velocity_x);
    auto* velocity_y_field = reinterpret_cast<float*>(desc->velocity_y);
    auto* velocity_z_field = reinterpret_cast<float*>(desc->velocity_z);
    auto* velocity_x_temporary = reinterpret_cast<float*>(desc->temporary_velocity_x);
    auto* velocity_y_temporary = reinterpret_cast<float*>(desc->temporary_velocity_y);
    auto* velocity_z_temporary = reinterpret_cast<float*>(desc->temporary_velocity_z);
    auto* velocity_x_previous = reinterpret_cast<float*>(desc->temporary_previous_velocity_x);
    auto* velocity_y_previous = reinterpret_cast<float*>(desc->temporary_previous_velocity_y);
    auto* velocity_z_previous = reinterpret_cast<float*>(desc->temporary_previous_velocity_z);
    auto* pressure = reinterpret_cast<float*>(desc->temporary_pressure);
    auto* divergence = reinterpret_cast<float*>(desc->temporary_divergence);
    const dim3 block{static_cast<unsigned>(std::max(block_x, 1)), static_cast<unsigned>(std::max(block_y, 1)), static_cast<unsigned>(std::max(block_z, 1))};
    const dim3 cells = make_grid(nx, ny, nz, block);
    const dim3 u_grid = make_grid(nx + 1, ny, nz, block);
    const dim3 v_grid = make_grid(nx, ny + 1, nz, block);
    const dim3 w_grid = make_grid(nx, ny, nz + 1, block);
    const auto stream = to_stream(desc->stream);

    nvtx3::scoped_range step_range{"stable.step"};
    {
        nvtx3::scoped_range range{"stable.step.advect_velocity"};
        if (cuda_code(cudaMemcpyAsync(velocity_x_previous, velocity_x_field, velocity_x_field_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
        if (cuda_code(cudaMemcpyAsync(velocity_y_previous, velocity_y_field, velocity_y_field_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
        if (cuda_code(cudaMemcpyAsync(velocity_z_previous, velocity_z_field, velocity_z_field_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
        advect_u_kernel<<<u_grid, block, 0, stream>>>(velocity_x_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, nx, ny, nz, cell_size, dt);
        advect_v_kernel<<<v_grid, block, 0, stream>>>(velocity_y_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, nx, ny, nz, cell_size, dt);
        advect_w_kernel<<<w_grid, block, 0, stream>>>(velocity_z_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, nx, ny, nz, cell_size, dt);
        set_u_boundary_kernel<<<u_grid, block, 0, stream>>>(velocity_x_temporary, nx, ny, nz);
        set_v_boundary_kernel<<<v_grid, block, 0, stream>>>(velocity_y_temporary, nx, ny, nz);
        set_w_boundary_kernel<<<w_grid, block, 0, stream>>>(velocity_z_temporary, nx, ny, nz);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    {
        nvtx3::scoped_range range{"stable.step.diffuse_velocity"};
        if (viscosity <= 0.0f) {
            if (cuda_code(cudaMemcpyAsync(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
            if (cuda_code(cudaMemcpyAsync(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
            if (cuda_code(cudaMemcpyAsync(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
        } else {
            if (cuda_code(cudaMemcpyAsync(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
            if (cuda_code(cudaMemcpyAsync(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
            if (cuda_code(cudaMemcpyAsync(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
            const float alpha = dt * viscosity / (cell_size * cell_size);
            const float denom = 1.0f + 6.0f * alpha;
            for (int iteration = 0; iteration < diffuse_iterations; ++iteration) {
                diffuse_grid_kernel<<<u_grid, block, 0, stream>>>(velocity_x_field, velocity_x_temporary, nx + 1, ny, nz, alpha, denom, 0);
                diffuse_grid_kernel<<<v_grid, block, 0, stream>>>(velocity_y_field, velocity_y_temporary, nx, ny + 1, nz, alpha, denom, 0);
                diffuse_grid_kernel<<<w_grid, block, 0, stream>>>(velocity_z_field, velocity_z_temporary, nx, ny, nz + 1, alpha, denom, 0);
                set_u_boundary_kernel<<<u_grid, block, 0, stream>>>(velocity_x_field, nx, ny, nz);
                set_v_boundary_kernel<<<v_grid, block, 0, stream>>>(velocity_y_field, nx, ny, nz);
                set_w_boundary_kernel<<<w_grid, block, 0, stream>>>(velocity_z_field, nx, ny, nz);
                if (cuda_code(cudaGetLastError()) != 0) return 5001;
                diffuse_grid_kernel<<<u_grid, block, 0, stream>>>(velocity_x_field, velocity_x_temporary, nx + 1, ny, nz, alpha, denom, 1);
                diffuse_grid_kernel<<<v_grid, block, 0, stream>>>(velocity_y_field, velocity_y_temporary, nx, ny + 1, nz, alpha, denom, 1);
                diffuse_grid_kernel<<<w_grid, block, 0, stream>>>(velocity_z_field, velocity_z_temporary, nx, ny, nz + 1, alpha, denom, 1);
                set_u_boundary_kernel<<<u_grid, block, 0, stream>>>(velocity_x_field, nx, ny, nz);
                set_v_boundary_kernel<<<v_grid, block, 0, stream>>>(velocity_y_field, nx, ny, nz);
                set_w_boundary_kernel<<<w_grid, block, 0, stream>>>(velocity_z_field, nx, ny, nz);
                if (cuda_code(cudaGetLastError()) != 0) return 5001;
            }
        }
    }
    {
        nvtx3::scoped_range range{"stable.step.project"};
        if (cuda_code(cudaMemsetAsync(pressure, 0, cell_bytes, stream)) != 0) return 5001;
        compute_divergence_kernel<<<cells, block, 0, stream>>>(divergence, velocity_x_field, velocity_y_field, velocity_z_field, nx, ny, nz, cell_size);
        for (int iteration = 0; iteration < pressure_iterations; ++iteration) {
            pressure_rbgs_kernel<<<cells, block, 0, stream>>>(pressure, divergence, nx, ny, nz, cell_size, 0);
            pressure_rbgs_kernel<<<cells, block, 0, stream>>>(pressure, divergence, nx, ny, nz, cell_size, 1);
        }
        subtract_gradient_u_kernel<<<u_grid, block, 0, stream>>>(velocity_x_field, pressure, nx, ny, nz, cell_size);
        subtract_gradient_v_kernel<<<v_grid, block, 0, stream>>>(velocity_y_field, pressure, nx, ny, nz, cell_size);
        subtract_gradient_w_kernel<<<w_grid, block, 0, stream>>>(velocity_z_field, pressure, nx, ny, nz, cell_size);
        set_u_boundary_kernel<<<u_grid, block, 0, stream>>>(velocity_x_field, nx, ny, nz);
        set_v_boundary_kernel<<<v_grid, block, 0, stream>>>(velocity_y_field, nx, ny, nz);
        set_w_boundary_kernel<<<w_grid, block, 0, stream>>>(velocity_z_field, nx, ny, nz);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    {
        nvtx3::scoped_range range{"stable.step.advect_density"};
        if (cuda_code(cudaMemcpyAsync(density_previous, density_field, cell_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
        advect_scalar_kernel<<<cells, block, 0, stream>>>(density_temporary, density_previous, velocity_x_field, velocity_y_field, velocity_z_field, nx, ny, nz, cell_size, dt);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    {
        nvtx3::scoped_range range{"stable.step.diffuse_density"};
        if (diffusion <= 0.0f) {
            if (cuda_code(cudaMemcpyAsync(density_field, density_temporary, cell_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
        } else {
            if (cuda_code(cudaMemcpyAsync(density_field, density_temporary, cell_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
            const float alpha = dt * diffusion / (cell_size * cell_size);
            const float denom = 1.0f + 6.0f * alpha;
            for (int iteration = 0; iteration < diffuse_iterations; ++iteration) {
                diffuse_grid_kernel<<<cells, block, 0, stream>>>(density_field, density_temporary, nx, ny, nz, alpha, denom, 0);
                if (cuda_code(cudaGetLastError()) != 0) return 5001;
                diffuse_grid_kernel<<<cells, block, 0, stream>>>(density_field, density_temporary, nx, ny, nz, alpha, denom, 1);
                if (cuda_code(cudaGetLastError()) != 0) return 5001;
            }
        }
    }
    return 0;
}

} // extern "C"
