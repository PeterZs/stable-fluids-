#include "stable-fluids.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace stable_fluids {
    using Stream = cudaStream_t;

    namespace {

        inline void check_cuda(cudaError_t status, const char* what) {
            if (status == cudaSuccess) {
                return;
            }
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        }

        inline std::uint64_t ghosted_bytes(const int32_t nx, const int32_t ny, const int32_t nz) {
            return static_cast<std::uint64_t>(nx + 2) * static_cast<std::uint64_t>(ny + 2) * static_cast<std::uint64_t>(nz + 2) * sizeof(float);
        }

        inline std::uint64_t workspace_bytes(const int32_t nx, const int32_t ny, const int32_t nz) {
            return ghosted_bytes(nx, ny, nz) * 10ull;
        }

        inline dim3 make_grid(int nx, int ny, int nz, const dim3& block) {
            return dim3(static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
        }

        __host__ __device__ inline int ghosted_index(int x, int y, int z, int nx_total, int ny_total) {
            return (z * ny_total + y) * nx_total + x;
        }

        __host__ __device__ inline std::uint64_t compact_index(int x, int y, int z, int nx, int ny) {
            return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nx) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(nx) + static_cast<std::uint64_t>(x);
        }

        __device__ inline float clampf(float value, float lo, float hi) {
            return fminf(fmaxf(value, lo), hi);
        }

        __device__ inline float sample_trilinear(const float* field, float x, float y, float z, int nx, int ny, int nz, int nx_total, int ny_total) {
            x = clampf(x, 0.5f, static_cast<float>(nx) + 0.5f);
            y = clampf(y, 0.5f, static_cast<float>(ny) + 0.5f);
            z = clampf(z, 0.5f, static_cast<float>(nz) + 0.5f);

            const int x0 = static_cast<int>(floorf(x));
            const int y0 = static_cast<int>(floorf(y));
            const int z0 = static_cast<int>(floorf(z));
            const int x1 = min(x0 + 1, nx + 1);
            const int y1 = min(y0 + 1, ny + 1);
            const int z1 = min(z0 + 1, nz + 1);

            const float tx = x - static_cast<float>(x0);
            const float ty = y - static_cast<float>(y0);
            const float tz = z - static_cast<float>(z0);

            const float c000 = field[ghosted_index(x0, y0, z0, nx_total, ny_total)];
            const float c100 = field[ghosted_index(x1, y0, z0, nx_total, ny_total)];
            const float c010 = field[ghosted_index(x0, y1, z0, nx_total, ny_total)];
            const float c110 = field[ghosted_index(x1, y1, z0, nx_total, ny_total)];
            const float c001 = field[ghosted_index(x0, y0, z1, nx_total, ny_total)];
            const float c101 = field[ghosted_index(x1, y0, z1, nx_total, ny_total)];
            const float c011 = field[ghosted_index(x0, y1, z1, nx_total, ny_total)];
            const float c111 = field[ghosted_index(x1, y1, z1, nx_total, ny_total)];

            const float c00 = c000 + tx * (c100 - c000);
            const float c10 = c010 + tx * (c110 - c010);
            const float c01 = c001 + tx * (c101 - c001);
            const float c11 = c011 + tx * (c111 - c011);

            const float c0 = c00 + ty * (c10 - c00);
            const float c1 = c01 + ty * (c11 - c01);
            return c0 + tz * (c1 - c0);
        }

        __global__ void fill_kernel(float* field, float value, std::size_t count) {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < count) {
                field[idx] = value;
            }
        }

        __global__ void zero_compact_field_kernel(float* field, std::size_t count) {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < count) {
                field[idx] = 0.0f;
            }
        }

        __global__ void copy_compact_to_ghosted_kernel(float* ghosted, const float* compact, int nx, int ny, int nx_total, int ny_total, std::size_t count) {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= count) {
                return;
            }

            const std::size_t plane                                         = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
            const int z                                                     = static_cast<int>(idx / plane);
            const std::size_t rem                                           = idx % plane;
            const int y                                                     = static_cast<int>(rem / static_cast<std::size_t>(nx));
            const int x                                                     = static_cast<int>(rem % static_cast<std::size_t>(nx));
            ghosted[ghosted_index(x + 1, y + 1, z + 1, nx_total, ny_total)] = compact[idx];
        }

        __global__ void copy_ghosted_to_compact_kernel(float* compact, const float* ghosted, int nx, int ny, int nx_total, int ny_total, std::size_t count) {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= count) {
                return;
            }

            const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
            const int z             = static_cast<int>(idx / plane);
            const std::size_t rem   = idx % plane;
            const int y             = static_cast<int>(rem / static_cast<std::size_t>(nx));
            const int x             = static_cast<int>(rem % static_cast<std::size_t>(nx));
            compact[idx]            = ghosted[ghosted_index(x + 1, y + 1, z + 1, nx_total, ny_total)];
        }

        __global__ void set_boundary_kernel(int field_kind, float* field, int nx, int ny, int nz, int nx_total, int ny_total) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx + 2 || j >= ny + 2 || k >= nz + 2) {
                return;
            }

            if (i > 0 && i < nx + 1 && j > 0 && j < ny + 1 && k > 0 && k < nz + 1) {
                return;
            }

            const int ii = min(max(i, 1), nx);
            const int jj = min(max(j, 1), ny);
            const int kk = min(max(k, 1), nz);

            float sign_sum = 0.0f;
            int sign_count = 0;
            if (i == 0 || i == nx + 1) {
                sign_sum += field_kind == 1 ? -1.0f : 1.0f;
                ++sign_count;
            }
            if (j == 0 || j == ny + 1) {
                sign_sum += field_kind == 2 ? -1.0f : 1.0f;
                ++sign_count;
            }
            if (k == 0 || k == nz + 1) {
                sign_sum += field_kind == 3 ? -1.0f : 1.0f;
                ++sign_count;
            }

            const int dst = ghosted_index(i, j, k, nx_total, ny_total);
            const int src = ghosted_index(ii, jj, kk, nx_total, ny_total);
            field[dst]    = field[src] * (sign_sum / static_cast<float>(max(sign_count, 1)));
        }

        __global__ void splat_density_compact_kernel(float* density, float center_x, float center_y, float center_z, float radius, float amount, int nx, int ny, int nz) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) {
                return;
            }

            const float dx      = static_cast<float>(x) - center_x;
            const float dy      = static_cast<float>(y) - center_y;
            const float dz      = static_cast<float>(z) - center_z;
            const float dist2   = dx * dx + dy * dy + dz * dz;
            const float radius2 = radius * radius;
            if (dist2 > radius2) {
                return;
            }

            const float falloff = 1.0f - sqrtf(dist2 / fmaxf(radius2, 1.0e-6f));
            density[compact_index(x, y, z, nx, ny)] += amount * falloff;
        }

        __global__ void splat_force_compact_kernel(float* u, float* v, float* w, float center_x, float center_y, float center_z, float radius, float fx, float fy, float fz, int nx, int ny, int nz) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) {
                return;
            }

            const float dx      = static_cast<float>(x) - center_x;
            const float dy      = static_cast<float>(y) - center_y;
            const float dz      = static_cast<float>(z) - center_z;
            const float dist2   = dx * dx + dy * dy + dz * dz;
            const float radius2 = radius * radius;
            if (dist2 > radius2) {
                return;
            }

            const float falloff     = 1.0f - sqrtf(dist2 / fmaxf(radius2, 1.0e-6f));
            const std::uint64_t idx = compact_index(x, y, z, nx, ny);
            u[idx] += fx * falloff;
            v[idx] += fy * falloff;
            w[idx] += fz * falloff;
        }

        __global__ void advect_scalar_kernel(float* dst, const float* src, const float* u, const float* v, const float* w, float dt_over_h, int nx, int ny, int nz, int nx_total, int ny_total) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) + 1;
            if (i > nx || j > ny || k > nz) {
                return;
            }

            const int idx      = ghosted_index(i, j, k, nx_total, ny_total);
            const float back_x = static_cast<float>(i) - dt_over_h * u[idx];
            const float back_y = static_cast<float>(j) - dt_over_h * v[idx];
            const float back_z = static_cast<float>(k) - dt_over_h * w[idx];
            dst[idx]           = sample_trilinear(src, back_x, back_y, back_z, nx, ny, nz, nx_total, ny_total);
        }

        __global__ void advect_velocity_kernel(float* u_dst, float* v_dst, float* w_dst, const float* u_src, const float* v_src, const float* w_src, const float* u_vel, const float* v_vel, const float* w_vel, float dt_over_h, int nx, int ny, int nz, int nx_total, int ny_total) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) + 1;
            if (i > nx || j > ny || k > nz) {
                return;
            }

            const int idx      = ghosted_index(i, j, k, nx_total, ny_total);
            const float back_x = static_cast<float>(i) - dt_over_h * u_vel[idx];
            const float back_y = static_cast<float>(j) - dt_over_h * v_vel[idx];
            const float back_z = static_cast<float>(k) - dt_over_h * w_vel[idx];

            u_dst[idx] = sample_trilinear(u_src, back_x, back_y, back_z, nx, ny, nz, nx_total, ny_total);
            v_dst[idx] = sample_trilinear(v_src, back_x, back_y, back_z, nx, ny, nz, nx_total, ny_total);
            w_dst[idx] = sample_trilinear(w_src, back_x, back_y, back_z, nx, ny, nz, nx_total, ny_total);
        }

        __global__ void rbgs_diffuse_kernel(float* dst, const float* src, float alpha, float denom, int parity, int nx, int ny, int nz, int nx_total, int ny_total) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) + 1;
            if (i > nx || j > ny || k > nz || ((i + j + k) & 1) != parity) {
                return;
            }

            const int idx         = ghosted_index(i, j, k, nx_total, ny_total);
            const float neighbors = dst[ghosted_index(i - 1, j, k, nx_total, ny_total)] + dst[ghosted_index(i + 1, j, k, nx_total, ny_total)] + dst[ghosted_index(i, j - 1, k, nx_total, ny_total)] + dst[ghosted_index(i, j + 1, k, nx_total, ny_total)] + dst[ghosted_index(i, j, k - 1, nx_total, ny_total)] + dst[ghosted_index(i, j, k + 1, nx_total, ny_total)];
            dst[idx]              = (src[idx] + alpha * neighbors) / denom;
        }

        __global__ void divergence_kernel(float* divergence, const float* u, const float* v, const float* w, float half_inv_h, int nx, int ny, int nz, int nx_total, int ny_total) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) + 1;
            if (i > nx || j > ny || k > nz) {
                return;
            }

            const int idx = ghosted_index(i, j, k, nx_total, ny_total);
            divergence[idx] =
                half_inv_h * ((u[ghosted_index(i + 1, j, k, nx_total, ny_total)] - u[ghosted_index(i - 1, j, k, nx_total, ny_total)]) + (v[ghosted_index(i, j + 1, k, nx_total, ny_total)] - v[ghosted_index(i, j - 1, k, nx_total, ny_total)]) + (w[ghosted_index(i, j, k + 1, nx_total, ny_total)] - w[ghosted_index(i, j, k - 1, nx_total, ny_total)]));
        }

        __global__ void rbgs_pressure_kernel(float* pressure, const float* divergence, float h2, int parity, int nx, int ny, int nz, int nx_total, int ny_total) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) + 1;
            if (i > nx || j > ny || k > nz || ((i + j + k) & 1) != parity) {
                return;
            }

            const int idx         = ghosted_index(i, j, k, nx_total, ny_total);
            const float neighbors = pressure[ghosted_index(i - 1, j, k, nx_total, ny_total)] +
                                    pressure[ghosted_index(i + 1, j, k, nx_total, ny_total)] +
                                    pressure[ghosted_index(i, j - 1, k, nx_total, ny_total)] +
                                    pressure[ghosted_index(i, j + 1, k, nx_total, ny_total)] +
                                    pressure[ghosted_index(i, j, k - 1, nx_total, ny_total)] +
                                    pressure[ghosted_index(i, j, k + 1, nx_total, ny_total)];
            pressure[idx]         = (neighbors - divergence[idx] * h2) / 6.0f;
        }

        __global__ void subtract_gradient_kernel(float* u, float* v, float* w, const float* pressure, float half_inv_h, int nx, int ny, int nz, int nx_total, int ny_total) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) + 1;
            if (i > nx || j > ny || k > nz) {
                return;
            }

            const int idx = ghosted_index(i, j, k, nx_total, ny_total);
            u[idx] -= half_inv_h * (pressure[ghosted_index(i + 1, j, k, nx_total, ny_total)] - pressure[ghosted_index(i - 1, j, k, nx_total, ny_total)]);
            v[idx] -= half_inv_h * (pressure[ghosted_index(i, j + 1, k, nx_total, ny_total)] - pressure[ghosted_index(i, j - 1, k, nx_total, ny_total)]);
            w[idx] -= half_inv_h * (pressure[ghosted_index(i, j, k + 1, nx_total, ny_total)] - pressure[ghosted_index(i, j, k - 1, nx_total, ny_total)]);
        }

        __global__ void velocity_magnitude_kernel(float* dst, const float* u, const float* v, const float* w, std::size_t count) {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < count) {
                const float ux = u[idx];
                const float vy = v[idx];
                const float wz = w[idx];
                dst[idx] = sqrtf(ux * ux + vy * vy + wz * wz);
            }
        }

    } // namespace

} // namespace stable_fluids

namespace {

    [[nodiscard]] stable_fluids::Stream to_stream(void* cuda_stream) noexcept {
        return reinterpret_cast<stable_fluids::Stream>(cuda_stream);
    }

} // namespace

extern "C" {

uint64_t stable_fluids_scalar_field_bytes(int32_t nx, int32_t ny, int32_t nz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        return 0;
    }
    return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float);
}

uint64_t stable_fluids_workspace_bytes(int32_t nx, int32_t ny, int32_t nz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        return 0;
    }
    return stable_fluids::workspace_bytes(nx, ny, nz);
}

int32_t stable_fluids_clear_async(
    void* density,
    uint64_t density_bytes,
    void* velocity_x,
    uint64_t velocity_x_bytes,
    void* velocity_y,
    uint64_t velocity_y_bytes,
    void* velocity_z,
    uint64_t velocity_z_bytes,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float cell_size,
    void* cuda_stream) {
    using namespace stable_fluids;
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("grid dimensions must be positive");
    }
    if (cell_size <= 0.0f) {
        throw std::invalid_argument("cell_size must be positive");
    }
    const auto compact_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    if (density == nullptr || density_bytes < compact_bytes) {
        throw std::invalid_argument("density is invalid");
    }
    if (velocity_x == nullptr || velocity_x_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_x is invalid");
    }
    if (velocity_y == nullptr || velocity_y_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_y is invalid");
    }
    if (velocity_z == nullptr || velocity_z_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_z is invalid");
    }

    nvtx3::scoped_range range{"stable.clear"};
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((compact_bytes / sizeof(float) + block_size - 1) / block_size);
    auto* density_f = reinterpret_cast<float*>(density);
    auto* u = reinterpret_cast<float*>(velocity_x);
    auto* v = reinterpret_cast<float*>(velocity_y);
    auto* w = reinterpret_cast<float*>(velocity_z);
    zero_compact_field_kernel<<<grid_size, block_size, 0, to_stream(cuda_stream)>>>(density_f, compact_bytes / sizeof(float));
    zero_compact_field_kernel<<<grid_size, block_size, 0, to_stream(cuda_stream)>>>(u, compact_bytes / sizeof(float));
    zero_compact_field_kernel<<<grid_size, block_size, 0, to_stream(cuda_stream)>>>(v, compact_bytes / sizeof(float));
    zero_compact_field_kernel<<<grid_size, block_size, 0, to_stream(cuda_stream)>>>(w, compact_bytes / sizeof(float));
    check_cuda(cudaGetLastError(), "zero_compact_field_kernel launch");
    return STABLE_FLUIDS_SUCCESS;
}

int32_t stable_fluids_add_density_splat_async(
    void* density,
    uint64_t density_bytes,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float cell_size,
    float center_x,
    float center_y,
    float center_z,
    float radius,
    float amount,
    void* cuda_stream) {
    using namespace stable_fluids;
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("grid dimensions must be positive");
    }
    if (cell_size <= 0.0f) {
        throw std::invalid_argument("cell_size must be positive");
    }
    if (density == nullptr) {
        throw std::invalid_argument("density must provide a non-null device pointer");
    }
    const auto compact_count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
    if (density_bytes < compact_count * sizeof(float)) {
        throw std::invalid_argument("density size_bytes is too small");
    }

    nvtx3::scoped_range range{"stable.add_density_splat"};
    const dim3 block{8u, 8u, 8u};
    const dim3 grid = make_grid(nx, ny, nz, block);
    splat_density_compact_kernel<<<grid, block, 0, to_stream(cuda_stream)>>>(
        reinterpret_cast<float*>(density),
        center_x,
        center_y,
        center_z,
        fmaxf(radius, 1.0f),
        amount,
        nx,
        ny,
        nz);
    check_cuda(cudaGetLastError(), "splat_density_compact_kernel launch");
    return STABLE_FLUIDS_SUCCESS;
}

int32_t stable_fluids_add_force_splat_async(
    void* velocity_x,
    uint64_t velocity_x_bytes,
    void* velocity_y,
    uint64_t velocity_y_bytes,
    void* velocity_z,
    uint64_t velocity_z_bytes,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float cell_size,
    float center_x,
    float center_y,
    float center_z,
    float radius,
    float force_x,
    float force_y,
    float force_z,
    void* cuda_stream) {
    using namespace stable_fluids;
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("grid dimensions must be positive");
    }
    if (cell_size <= 0.0f) {
        throw std::invalid_argument("cell_size must be positive");
    }
    const auto compact_count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
    const auto compact_bytes = compact_count * sizeof(float);
    if (velocity_x == nullptr || velocity_x_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_x is invalid");
    }
    if (velocity_y == nullptr || velocity_y_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_y is invalid");
    }
    if (velocity_z == nullptr || velocity_z_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_z is invalid");
    }

    nvtx3::scoped_range range{"stable.add_force_splat"};
    const dim3 block{8u, 8u, 8u};
    const dim3 grid = make_grid(nx, ny, nz, block);
    splat_force_compact_kernel<<<grid, block, 0, to_stream(cuda_stream)>>>(
        reinterpret_cast<float*>(velocity_x),
        reinterpret_cast<float*>(velocity_y),
        reinterpret_cast<float*>(velocity_z),
        center_x,
        center_y,
        center_z,
        fmaxf(radius, 1.0f),
        force_x,
        force_y,
        force_z,
        nx,
        ny,
        nz);
    check_cuda(cudaGetLastError(), "splat_force_compact_kernel launch");
    return STABLE_FLUIDS_SUCCESS;
}

int32_t stable_fluids_step_async(
    void* density,
    uint64_t density_bytes,
    void* velocity_x,
    uint64_t velocity_x_bytes,
    void* velocity_y,
    uint64_t velocity_y_bytes,
    void* velocity_z,
    uint64_t velocity_z_bytes,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float cell_size,
    void* workspace,
    uint64_t workspace_bytes,
    float dt,
    float viscosity,
    float diffusion,
    int32_t diffuse_iterations,
    int32_t pressure_iterations,
    int32_t block_x,
    int32_t block_y,
    int32_t block_z,
    void* cuda_stream) {
    using namespace stable_fluids;
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("grid dimensions must be positive");
    }
    if (cell_size <= 0.0f) {
        throw std::invalid_argument("cell_size must be positive");
    }
    if (dt <= 0.0f) {
        throw std::invalid_argument("dt must be positive");
    }
    if (diffuse_iterations <= 0 || pressure_iterations <= 0) {
        throw std::invalid_argument("iteration counts must be positive");
    }

    const auto interior_cell_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    const auto compact_bytes = interior_cell_count * sizeof(float);
    const int nx_total = nx + 2;
    const int ny_total = ny + 2;
    const int nz_total = nz + 2;
    const auto total_cell_count = static_cast<std::size_t>(nx_total) * static_cast<std::size_t>(ny_total) * static_cast<std::size_t>(nz_total);
    const auto ghosted_field_bytes = total_cell_count * sizeof(float);

    if (density == nullptr || density_bytes < compact_bytes) {
        throw std::invalid_argument("density is invalid");
    }
    if (velocity_x == nullptr || velocity_x_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_x is invalid");
    }
    if (velocity_y == nullptr || velocity_y_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_y is invalid");
    }
    if (velocity_z == nullptr || velocity_z_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_z is invalid");
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("workspace must provide a non-null device pointer");
    }
    if (workspace_bytes < ghosted_field_bytes * 10ull) {
        throw std::invalid_argument("workspace size_bytes is too small");
    }

    auto* cursor = reinterpret_cast<std::byte*>(workspace);
    auto* density_g = reinterpret_cast<float*>(cursor);
    cursor += ghosted_field_bytes;
    auto* u_g = reinterpret_cast<float*>(cursor);
    cursor += ghosted_field_bytes;
    auto* v_g = reinterpret_cast<float*>(cursor);
    cursor += ghosted_field_bytes;
    auto* w_g = reinterpret_cast<float*>(cursor);
    cursor += ghosted_field_bytes;
    auto* density_prev = reinterpret_cast<float*>(cursor);
    cursor += ghosted_field_bytes;
    auto* u_prev = reinterpret_cast<float*>(cursor);
    cursor += ghosted_field_bytes;
    auto* v_prev = reinterpret_cast<float*>(cursor);
    cursor += ghosted_field_bytes;
    auto* w_prev = reinterpret_cast<float*>(cursor);
    cursor += ghosted_field_bytes;
    auto* pressure = reinterpret_cast<float*>(cursor);
    cursor += ghosted_field_bytes;
    auto* divergence = reinterpret_cast<float*>(cursor);
    auto* density_compact = reinterpret_cast<float*>(density);
    auto* u_compact = reinterpret_cast<float*>(velocity_x);
    auto* v_compact = reinterpret_cast<float*>(velocity_y);
    auto* w_compact = reinterpret_cast<float*>(velocity_z);
    const dim3 block{
        static_cast<unsigned>(std::max(block_x, 1)),
        static_cast<unsigned>(std::max(block_y, 1)),
        static_cast<unsigned>(std::max(block_z, 1))};
    const dim3 interior_grid = make_grid(nx, ny, nz, block);
    const dim3 ghost_grid = make_grid(nx_total, ny_total, nz_total, block);
    const auto stream = to_stream(cuda_stream);
    constexpr int linear_block_size = 256;
    const int fill_grid = static_cast<int>((total_cell_count + linear_block_size - 1) / linear_block_size);
    const int compact_grid = static_cast<int>((interior_cell_count + linear_block_size - 1) / linear_block_size);
    const float dt_over_h = dt / cell_size;

    nvtx3::scoped_range step_range{"stable.step"};
    fill_kernel<<<fill_grid, linear_block_size, 0, stream>>>(density_g, 0.0f, total_cell_count);
    fill_kernel<<<fill_grid, linear_block_size, 0, stream>>>(u_g, 0.0f, total_cell_count);
    fill_kernel<<<fill_grid, linear_block_size, 0, stream>>>(v_g, 0.0f, total_cell_count);
    fill_kernel<<<fill_grid, linear_block_size, 0, stream>>>(w_g, 0.0f, total_cell_count);
    check_cuda(cudaGetLastError(), "fill_kernel launch");
    copy_compact_to_ghosted_kernel<<<compact_grid, linear_block_size, 0, stream>>>(density_g, density_compact, nx, ny, nx_total, ny_total, interior_cell_count);
    copy_compact_to_ghosted_kernel<<<compact_grid, linear_block_size, 0, stream>>>(u_g, u_compact, nx, ny, nx_total, ny_total, interior_cell_count);
    copy_compact_to_ghosted_kernel<<<compact_grid, linear_block_size, 0, stream>>>(v_g, v_compact, nx, ny, nx_total, ny_total, interior_cell_count);
    copy_compact_to_ghosted_kernel<<<compact_grid, linear_block_size, 0, stream>>>(w_g, w_compact, nx, ny, nx_total, ny_total, interior_cell_count);
    check_cuda(cudaGetLastError(), "copy_compact_to_ghosted_kernel launch");
    set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(0, density_g, nx, ny, nz, nx_total, ny_total);
    set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(1, u_g, nx, ny, nz, nx_total, ny_total);
    set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(2, v_g, nx, ny, nz, nx_total, ny_total);
    set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(3, w_g, nx, ny, nz, nx_total, ny_total);
    check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
    {
        nvtx3::scoped_range range{"stable.step.advect_velocity"};
        advect_velocity_kernel<<<interior_grid, block, 0, stream>>>(u_prev, v_prev, w_prev, u_g, v_g, w_g, u_g, v_g, w_g, dt_over_h, nx, ny, nz, nx_total, ny_total);
        check_cuda(cudaGetLastError(), "advect_velocity_kernel launch");
        set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(1, u_prev, nx, ny, nz, nx_total, ny_total);
        set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(2, v_prev, nx, ny, nz, nx_total, ny_total);
        set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(3, w_prev, nx, ny, nz, nx_total, ny_total);
        check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
    }
    {
        nvtx3::scoped_range range{"stable.step.diffuse_velocity"};
        if (viscosity <= 0.0f) {
            check_cuda(cudaMemcpyAsync(u_g, u_prev, ghosted_field_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync u diffuse bypass");
            check_cuda(cudaMemcpyAsync(v_g, v_prev, ghosted_field_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync v diffuse bypass");
            check_cuda(cudaMemcpyAsync(w_g, w_prev, ghosted_field_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync w diffuse bypass");
            set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(1, u_g, nx, ny, nz, nx_total, ny_total);
            set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(2, v_g, nx, ny, nz, nx_total, ny_total);
            set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(3, w_g, nx, ny, nz, nx_total, ny_total);
            check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
        } else {
            check_cuda(cudaMemcpyAsync(u_g, u_prev, ghosted_field_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync u diffuse init");
            check_cuda(cudaMemcpyAsync(v_g, v_prev, ghosted_field_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync v diffuse init");
            check_cuda(cudaMemcpyAsync(w_g, w_prev, ghosted_field_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync w diffuse init");
            const float alpha = dt * viscosity / (cell_size * cell_size);
            const float denom = 1.0f + 6.0f * alpha;
            set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(1, u_g, nx, ny, nz, nx_total, ny_total);
            set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(2, v_g, nx, ny, nz, nx_total, ny_total);
            set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(3, w_g, nx, ny, nz, nx_total, ny_total);
            check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
            for (int iteration = 0; iteration < diffuse_iterations; ++iteration) {
                rbgs_diffuse_kernel<<<interior_grid, block, 0, stream>>>(u_g, u_prev, alpha, denom, 0, nx, ny, nz, nx_total, ny_total);
                rbgs_diffuse_kernel<<<interior_grid, block, 0, stream>>>(v_g, v_prev, alpha, denom, 0, nx, ny, nz, nx_total, ny_total);
                rbgs_diffuse_kernel<<<interior_grid, block, 0, stream>>>(w_g, w_prev, alpha, denom, 0, nx, ny, nz, nx_total, ny_total);
                check_cuda(cudaGetLastError(), "rbgs_diffuse_kernel velocity red launch");
                set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(1, u_g, nx, ny, nz, nx_total, ny_total);
                set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(2, v_g, nx, ny, nz, nx_total, ny_total);
                set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(3, w_g, nx, ny, nz, nx_total, ny_total);
                check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
                rbgs_diffuse_kernel<<<interior_grid, block, 0, stream>>>(u_g, u_prev, alpha, denom, 1, nx, ny, nz, nx_total, ny_total);
                rbgs_diffuse_kernel<<<interior_grid, block, 0, stream>>>(v_g, v_prev, alpha, denom, 1, nx, ny, nz, nx_total, ny_total);
                rbgs_diffuse_kernel<<<interior_grid, block, 0, stream>>>(w_g, w_prev, alpha, denom, 1, nx, ny, nz, nx_total, ny_total);
                check_cuda(cudaGetLastError(), "rbgs_diffuse_kernel velocity black launch");
                set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(1, u_g, nx, ny, nz, nx_total, ny_total);
                set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(2, v_g, nx, ny, nz, nx_total, ny_total);
                set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(3, w_g, nx, ny, nz, nx_total, ny_total);
                check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
            }
        }
    }
    {
        nvtx3::scoped_range range{"stable.step.project"};
        const float half_inv_h = 0.5f / cell_size;
        const float h2 = cell_size * cell_size;
        divergence_kernel<<<interior_grid, block, 0, stream>>>(divergence, u_g, v_g, w_g, half_inv_h, nx, ny, nz, nx_total, ny_total);
        check_cuda(cudaGetLastError(), "divergence_kernel launch");
        set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(0, divergence, nx, ny, nz, nx_total, ny_total);
        check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
        fill_kernel<<<fill_grid, linear_block_size, 0, stream>>>(pressure, 0.0f, total_cell_count);
        check_cuda(cudaGetLastError(), "fill_kernel launch");
        set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(0, pressure, nx, ny, nz, nx_total, ny_total);
        check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
        for (int iteration = 0; iteration < pressure_iterations; ++iteration) {
            rbgs_pressure_kernel<<<interior_grid, block, 0, stream>>>(pressure, divergence, h2, 0, nx, ny, nz, nx_total, ny_total);
            check_cuda(cudaGetLastError(), "rbgs_pressure_kernel red launch");
            set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(0, pressure, nx, ny, nz, nx_total, ny_total);
            check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
            rbgs_pressure_kernel<<<interior_grid, block, 0, stream>>>(pressure, divergence, h2, 1, nx, ny, nz, nx_total, ny_total);
            check_cuda(cudaGetLastError(), "rbgs_pressure_kernel black launch");
            set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(0, pressure, nx, ny, nz, nx_total, ny_total);
            check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
        }
        subtract_gradient_kernel<<<interior_grid, block, 0, stream>>>(u_g, v_g, w_g, pressure, half_inv_h, nx, ny, nz, nx_total, ny_total);
        check_cuda(cudaGetLastError(), "subtract_gradient_kernel launch");
        set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(1, u_g, nx, ny, nz, nx_total, ny_total);
        set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(2, v_g, nx, ny, nz, nx_total, ny_total);
        set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(3, w_g, nx, ny, nz, nx_total, ny_total);
        check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
    }
    {
        nvtx3::scoped_range range{"stable.step.advect_density"};
        advect_scalar_kernel<<<interior_grid, block, 0, stream>>>(density_prev, density_g, u_g, v_g, w_g, dt_over_h, nx, ny, nz, nx_total, ny_total);
        check_cuda(cudaGetLastError(), "advect_scalar_kernel launch");
        set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(0, density_prev, nx, ny, nz, nx_total, ny_total);
        check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
    }
    {
        nvtx3::scoped_range range{"stable.step.diffuse_density"};
        if (diffusion <= 0.0f) {
            check_cuda(cudaMemcpyAsync(density_g, density_prev, ghosted_field_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync scalar diffuse bypass");
            set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(0, density_g, nx, ny, nz, nx_total, ny_total);
            check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
        } else {
            check_cuda(cudaMemcpyAsync(density_g, density_prev, ghosted_field_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync scalar diffuse init");
            const float alpha = dt * diffusion / (cell_size * cell_size);
            const float denom = 1.0f + 6.0f * alpha;
            set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(0, density_g, nx, ny, nz, nx_total, ny_total);
            check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
            for (int iteration = 0; iteration < diffuse_iterations; ++iteration) {
                rbgs_diffuse_kernel<<<interior_grid, block, 0, stream>>>(density_g, density_prev, alpha, denom, 0, nx, ny, nz, nx_total, ny_total);
                check_cuda(cudaGetLastError(), "rbgs_diffuse_kernel scalar red launch");
                set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(0, density_g, nx, ny, nz, nx_total, ny_total);
                check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
                rbgs_diffuse_kernel<<<interior_grid, block, 0, stream>>>(density_g, density_prev, alpha, denom, 1, nx, ny, nz, nx_total, ny_total);
                check_cuda(cudaGetLastError(), "rbgs_diffuse_kernel scalar black launch");
                set_boundary_kernel<<<ghost_grid, block, 0, stream>>>(0, density_g, nx, ny, nz, nx_total, ny_total);
                check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
            }
        }
    }
    copy_ghosted_to_compact_kernel<<<compact_grid, linear_block_size, 0, stream>>>(density_compact, density_g, nx, ny, nx_total, ny_total, interior_cell_count);
    copy_ghosted_to_compact_kernel<<<compact_grid, linear_block_size, 0, stream>>>(u_compact, u_g, nx, ny, nx_total, ny_total, interior_cell_count);
    copy_ghosted_to_compact_kernel<<<compact_grid, linear_block_size, 0, stream>>>(v_compact, v_g, nx, ny, nx_total, ny_total, interior_cell_count);
    copy_ghosted_to_compact_kernel<<<compact_grid, linear_block_size, 0, stream>>>(w_compact, w_g, nx, ny, nx_total, ny_total, interior_cell_count);
    check_cuda(cudaGetLastError(), "copy_ghosted_to_compact_kernel launch");
    return STABLE_FLUIDS_SUCCESS;
}

int32_t stable_fluids_compute_velocity_magnitude_async(
    void* velocity_x,
    uint64_t velocity_x_bytes,
    void* velocity_y,
    uint64_t velocity_y_bytes,
    void* velocity_z,
    uint64_t velocity_z_bytes,
    void* destination,
    uint64_t destination_bytes,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float cell_size,
    void* cuda_stream) {
    using namespace stable_fluids;
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("grid dimensions must be positive");
    }
    if (cell_size <= 0.0f) {
        throw std::invalid_argument("cell_size must be positive");
    }
    const auto compact_count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
    const auto compact_bytes = compact_count * sizeof(float);
    if (velocity_x == nullptr || velocity_x_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_x is invalid");
    }
    if (velocity_y == nullptr || velocity_y_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_y is invalid");
    }
    if (velocity_z == nullptr || velocity_z_bytes < compact_bytes) {
        throw std::invalid_argument("velocity_z is invalid");
    }
    if (destination == nullptr || destination_bytes < compact_bytes) {
        throw std::invalid_argument("destination is invalid");
    }

    nvtx3::scoped_range range{"stable.snapshot_velocity_magnitude"};
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((compact_count + block_size - 1) / block_size);
    velocity_magnitude_kernel<<<grid_size, block_size, 0, to_stream(cuda_stream)>>>(
        reinterpret_cast<float*>(destination),
        reinterpret_cast<const float*>(velocity_x),
        reinterpret_cast<const float*>(velocity_y),
        reinterpret_cast<const float*>(velocity_z),
        compact_count);
    check_cuda(cudaGetLastError(), "velocity_magnitude_kernel launch");
    return STABLE_FLUIDS_SUCCESS;
}

} // extern "C"
