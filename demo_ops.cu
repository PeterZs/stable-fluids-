#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

    int32_t cuda_code(const cudaError_t status) noexcept {
        return status == cudaSuccess ? 0 : 5001;
    }

    dim3 make_grid(const int nx, const int ny, const int nz, const dim3& block) {
        return dim3(static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
    }

    __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
    }

    __global__ void density_splat_kernel(float* density, const float center_x, const float center_y, const float center_z, const float radius, const float amount, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const float dx = static_cast<float>(x) - center_x;
        const float dy = static_cast<float>(y) - center_y;
        const float dz = static_cast<float>(z) - center_z;
        const float radius2 = radius * radius;
        const float dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 > radius2) return;
        density[index_3d(x, y, z, nx, ny)] += amount * (1.0f - sqrtf(dist2 / fmaxf(radius2, 1.0e-6f)));
    }

    __global__ void force_splat_kernel(float* velocity_x, float* velocity_y, float* velocity_z, const float center_x, const float center_y, const float center_z, const float radius, const float force_x, const float force_y, const float force_z, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const float dx = static_cast<float>(x) - center_x;
        const float dy = static_cast<float>(y) - center_y;
        const float dz = static_cast<float>(z) - center_z;
        const float radius2 = radius * radius;
        const float dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 > radius2) return;

        const auto index = index_3d(x, y, z, nx, ny);
        const float weight = 1.0f - sqrtf(dist2 / fmaxf(radius2, 1.0e-6f));
        velocity_x[index] += force_x * weight;
        velocity_y[index] += force_y * weight;
        velocity_z[index] += force_z * weight;
    }

} // namespace

int32_t stable_demo_add_density_splat_async(void* density, int32_t nx, int32_t ny, int32_t nz, float center_x, float center_y, float center_z, float radius, float amount, int32_t block_x, int32_t block_y, int32_t block_z, void* cuda_stream) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return 1001;
    if (radius <= 0.0f) return 1005;
    if (density == nullptr) return 2001;

    const dim3 block{static_cast<unsigned>(std::max(block_x, 1)), static_cast<unsigned>(std::max(block_y, 1)), static_cast<unsigned>(std::max(block_z, 1))};
    density_splat_kernel<<<make_grid(nx, ny, nz, block), block, 0, reinterpret_cast<cudaStream_t>(cuda_stream)>>>(reinterpret_cast<float*>(density), center_x, center_y, center_z, radius, amount, nx, ny, nz);
    return cuda_code(cudaGetLastError());
}

int32_t stable_demo_add_force_splat_async(void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, float center_x, float center_y, float center_z, float radius, float force_x, float force_y, float force_z, int32_t block_x, int32_t block_y, int32_t block_z, void* cuda_stream) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return 1001;
    if (radius <= 0.0f) return 1005;
    if (velocity_x == nullptr) return 2001;
    if (velocity_y == nullptr) return 2002;
    if (velocity_z == nullptr) return 2003;

    const dim3 block{static_cast<unsigned>(std::max(block_x, 1)), static_cast<unsigned>(std::max(block_y, 1)), static_cast<unsigned>(std::max(block_z, 1))};
    force_splat_kernel<<<make_grid(nx, ny, nz, block), block, 0, reinterpret_cast<cudaStream_t>(cuda_stream)>>>(reinterpret_cast<float*>(velocity_x), reinterpret_cast<float*>(velocity_y), reinterpret_cast<float*>(velocity_z), center_x, center_y, center_z, radius, force_x, force_y, force_z, nx, ny, nz);
    return cuda_code(cudaGetLastError());
}
