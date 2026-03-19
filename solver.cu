#include "stable-fluids.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace stable_fluids {
    using SolverDesc   = StableFluidsContextDesc;
    using BufferView   = FieldBufferView;
    using ScalarFieldT = ScalarField;
    using VectorFieldT = VectorField;
    using FieldSet     = StableFluidsFieldSet;
    using DensitySplat = StableFluidsDensitySplatDesc;
    using ForceSplat   = StableFluidsForceSplatDesc;
    using Stream       = cudaStream_t;

    namespace {

        thread_local std::string g_last_error;

        inline void check_cuda(cudaError_t status, const char* what) {
            if (status == cudaSuccess) {
                return;
            }
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        }

        inline dim3 make_block(const SolverDesc& desc) {
            return dim3(static_cast<unsigned>(std::max(desc.block_x, 1)), static_cast<unsigned>(std::max(desc.block_y, 1)), static_cast<unsigned>(std::max(desc.block_z, 1)));
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

    class Solver3D {
    public:
        explicit Solver3D(const SolverDesc& desc);
        ~Solver3D();

        Solver3D(const Solver3D&)            = delete;
        Solver3D& operator=(const Solver3D&) = delete;
        Solver3D(Solver3D&&)                 = delete;
        Solver3D& operator=(Solver3D&&)      = delete;

        [[nodiscard]] std::uint64_t required_elements() const noexcept;
        [[nodiscard]] std::uint64_t required_scalar_field_bytes() const noexcept;
        [[nodiscard]] std::uint64_t required_vector_field_component_bytes(uint32_t component) const noexcept;

        void zero_fields(const FieldSet& fields, Stream stream);
        void add_density_sphere(const FieldSet& fields, float x, float y, float z, float radius, float amount, Stream stream);
        void add_force_sphere(const FieldSet& fields, float x, float y, float z, float radius, float fx, float fy, float fz, Stream stream);
        void step(const FieldSet& fields, Stream stream);
        void snapshot_density(const FieldSet& fields, const ScalarFieldT& destination, Stream stream);
        void snapshot_velocity_magnitude(const FieldSet& fields, const ScalarFieldT& destination, Stream stream);

    private:
        void validate_fields_(const FieldSet& fields) const;
        void fill_field_(float* field, float value, Stream stream);
        void zero_compact_field_(float* field, Stream stream);
        void copy_compact_to_ghosted_(float* ghosted, const float* compact, Stream stream);
        void copy_ghosted_to_compact_(float* compact, const float* ghosted, Stream stream);
        void set_boundary_(int field_kind, float* field, Stream stream);
        void advect_scalar_(float* dst, const float* src, const float* u, const float* v, const float* w, Stream stream);
        void advect_velocity_(float* u_dst, float* v_dst, float* w_dst, const float* u_src, const float* v_src, const float* w_src, const float* u_vel, const float* v_vel, const float* w_vel, Stream stream);
        void diffuse_scalar_(float* dst, const float* src, Stream stream);
        void diffuse_velocity_(float* u_dst, float* v_dst, float* w_dst, const float* u_src, const float* v_src, const float* w_src, Stream stream);
        void project_(float* u, float* v, float* w, Stream stream);

        SolverDesc desc_{};
        std::size_t interior_cell_count_ = 0;
        std::size_t total_cell_count_    = 0;
        std::size_t compact_bytes_       = 0;
        std::size_t ghosted_bytes_       = 0;
        int nx_total_                    = 0;
        int ny_total_                    = 0;
        int nz_total_                    = 0;

        float* density_      = nullptr;
        float* u_            = nullptr;
        float* v_            = nullptr;
        float* w_            = nullptr;
        float* density_prev_ = nullptr;
        float* u_prev_       = nullptr;
        float* v_prev_       = nullptr;
        float* w_prev_       = nullptr;
        float* pressure_     = nullptr;
        float* divergence_   = nullptr;
    };

    Solver3D::Solver3D(const SolverDesc& desc) : desc_(desc) {
        if (desc_.nx <= 0 || desc_.ny <= 0 || desc_.nz <= 0) {
            throw std::invalid_argument("grid dimensions must be positive");
        }
        if (desc_.dt <= 0.0f || desc_.cell_size <= 0.0f) {
            throw std::invalid_argument("dt and cell_size must be positive");
        }
        if (desc_.diffuse_iterations <= 0 || desc_.pressure_iterations <= 0) {
            throw std::invalid_argument("iteration counts must be positive");
        }

        nx_total_            = desc_.nx + 2;
        ny_total_            = desc_.ny + 2;
        nz_total_            = desc_.nz + 2;
        interior_cell_count_ = static_cast<std::size_t>(desc_.nx) * static_cast<std::size_t>(desc_.ny) * static_cast<std::size_t>(desc_.nz);
        total_cell_count_    = static_cast<std::size_t>(nx_total_) * static_cast<std::size_t>(ny_total_) * static_cast<std::size_t>(nz_total_);
        compact_bytes_       = interior_cell_count_ * sizeof(float);
        ghosted_bytes_       = total_cell_count_ * sizeof(float);

        auto allocate = [this](float*& field) { check_cuda(cudaMalloc(reinterpret_cast<void**>(&field), ghosted_bytes_), "cudaMalloc ghosted field"); };

        allocate(density_);
        allocate(u_);
        allocate(v_);
        allocate(w_);
        allocate(density_prev_);
        allocate(u_prev_);
        allocate(v_prev_);
        allocate(w_prev_);
        allocate(pressure_);
        allocate(divergence_);

        fill_field_(density_, 0.0f, nullptr);
        fill_field_(u_, 0.0f, nullptr);
        fill_field_(v_, 0.0f, nullptr);
        fill_field_(w_, 0.0f, nullptr);
        fill_field_(density_prev_, 0.0f, nullptr);
        fill_field_(u_prev_, 0.0f, nullptr);
        fill_field_(v_prev_, 0.0f, nullptr);
        fill_field_(w_prev_, 0.0f, nullptr);
        fill_field_(pressure_, 0.0f, nullptr);
        fill_field_(divergence_, 0.0f, nullptr);
        check_cuda(cudaDeviceSynchronize(), "constructor synchronize");
    }

    Solver3D::~Solver3D() {
        cudaFree(density_);
        cudaFree(u_);
        cudaFree(v_);
        cudaFree(w_);
        cudaFree(density_prev_);
        cudaFree(u_prev_);
        cudaFree(v_prev_);
        cudaFree(w_prev_);
        cudaFree(pressure_);
        cudaFree(divergence_);
    }

    std::uint64_t Solver3D::required_elements() const noexcept {
        return static_cast<std::uint64_t>(interior_cell_count_);
    }

    std::uint64_t Solver3D::required_scalar_field_bytes() const noexcept {
        return static_cast<std::uint64_t>(compact_bytes_);
    }

    std::uint64_t Solver3D::required_vector_field_component_bytes(uint32_t component) const noexcept {
        return component <= VECTOR_FIELD_COMPONENT_Z ? static_cast<std::uint64_t>(compact_bytes_) : 0;
    }

    void Solver3D::validate_fields_(const FieldSet& fields) const {
        const auto validate_grid = [this](const FieldGridDesc& grid, const char* name) {
            if (grid.nx != desc_.nx || grid.ny != desc_.ny || grid.nz != desc_.nz) {
                throw std::invalid_argument(std::string(name) + " grid dimensions must match the context");
            }
            if (std::fabs(grid.cell_size - desc_.cell_size) > 1.0e-6f) {
                throw std::invalid_argument(std::string(name) + " cell_size must match the context");
            }
        };

        const auto validate_buffer = [this](const BufferView& view, const char* name) {
            if (view.data == nullptr) {
                throw std::invalid_argument(std::string(name) + " must provide a non-null device pointer");
            }
            if (view.format != FIELD_FORMAT_F32) {
                throw std::invalid_argument(std::string(name) + " must use FIELD_FORMAT_F32");
            }
            if (view.memory_type != FIELD_MEMORY_TYPE_CUDA_DEVICE) {
                throw std::invalid_argument(std::string(name) + " must use FIELD_MEMORY_TYPE_CUDA_DEVICE");
            }
            if (view.size_bytes < static_cast<std::uint64_t>(compact_bytes_)) {
                throw std::invalid_argument(std::string(name) + " size_bytes is smaller than nx*ny*nz*sizeof(float)");
            }
        };

        validate_grid(fields.density.grid, "density");
        validate_buffer(fields.density.values, "density.values");

        validate_grid(fields.velocity.grid, "velocity");
        if (fields.velocity.layout != VECTOR_FIELD_LAYOUT_CELL_CENTERED) {
            throw std::invalid_argument("velocity.layout must be VECTOR_FIELD_LAYOUT_CELL_CENTERED");
        }
        validate_buffer(fields.velocity.x, "velocity.x");
        validate_buffer(fields.velocity.y, "velocity.y");
        validate_buffer(fields.velocity.z, "velocity.z");
    }

    void Solver3D::fill_field_(float* field, float value, Stream stream) {
        constexpr int block_size = 256;
        const int grid_size      = static_cast<int>((total_cell_count_ + block_size - 1) / block_size);
        fill_kernel<<<grid_size, block_size, 0, stream>>>(field, value, total_cell_count_);
        check_cuda(cudaGetLastError(), "fill_kernel launch");
    }

    void Solver3D::zero_compact_field_(float* field, Stream stream) {
        constexpr int block_size = 256;
        const int grid_size      = static_cast<int>((interior_cell_count_ + block_size - 1) / block_size);
        zero_compact_field_kernel<<<grid_size, block_size, 0, stream>>>(field, interior_cell_count_);
        check_cuda(cudaGetLastError(), "zero_compact_field_kernel launch");
    }

    void Solver3D::copy_compact_to_ghosted_(float* ghosted, const float* compact, Stream stream) {
        constexpr int block_size = 256;
        const int grid_size      = static_cast<int>((interior_cell_count_ + block_size - 1) / block_size);
        copy_compact_to_ghosted_kernel<<<grid_size, block_size, 0, stream>>>(ghosted, compact, desc_.nx, desc_.ny, nx_total_, ny_total_, interior_cell_count_);
        check_cuda(cudaGetLastError(), "copy_compact_to_ghosted_kernel launch");
    }

    void Solver3D::copy_ghosted_to_compact_(float* compact, const float* ghosted, Stream stream) {
        constexpr int block_size = 256;
        const int grid_size      = static_cast<int>((interior_cell_count_ + block_size - 1) / block_size);
        copy_ghosted_to_compact_kernel<<<grid_size, block_size, 0, stream>>>(compact, ghosted, desc_.nx, desc_.ny, nx_total_, ny_total_, interior_cell_count_);
        check_cuda(cudaGetLastError(), "copy_ghosted_to_compact_kernel launch");
    }

    void Solver3D::set_boundary_(int field_kind, float* field, Stream stream) {
        const dim3 block = make_block(desc_);
        const dim3 grid  = make_grid(nx_total_, ny_total_, nz_total_, block);
        set_boundary_kernel<<<grid, block, 0, stream>>>(field_kind, field, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
        check_cuda(cudaGetLastError(), "set_boundary_kernel launch");
    }

    void Solver3D::zero_fields(const FieldSet& fields, Stream stream) {
        nvtx3::scoped_range range{"stable.clear"};
        validate_fields_(fields);
        zero_compact_field_(reinterpret_cast<float*>(fields.density.values.data), stream);
        zero_compact_field_(reinterpret_cast<float*>(fields.velocity.x.data), stream);
        zero_compact_field_(reinterpret_cast<float*>(fields.velocity.y.data), stream);
        zero_compact_field_(reinterpret_cast<float*>(fields.velocity.z.data), stream);
    }

    void Solver3D::add_density_sphere(const FieldSet& fields, float x, float y, float z, float radius, float amount, Stream stream) {
        nvtx3::scoped_range range{"stable.add_density_splat"};
        validate_fields_(fields);
        const dim3 block = make_block(desc_);
        const dim3 grid  = make_grid(desc_.nx, desc_.ny, desc_.nz, block);
        splat_density_compact_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<float*>(fields.density.values.data), x, y, z, fmaxf(radius, 1.0f), amount, desc_.nx, desc_.ny, desc_.nz);
        check_cuda(cudaGetLastError(), "splat_density_compact_kernel launch");
    }

    void Solver3D::add_force_sphere(const FieldSet& fields, float x, float y, float z, float radius, float fx, float fy, float fz, Stream stream) {
        nvtx3::scoped_range range{"stable.add_force_splat"};
        validate_fields_(fields);
        const dim3 block = make_block(desc_);
        const dim3 grid  = make_grid(desc_.nx, desc_.ny, desc_.nz, block);
        splat_force_compact_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<float*>(fields.velocity.x.data), reinterpret_cast<float*>(fields.velocity.y.data), reinterpret_cast<float*>(fields.velocity.z.data), x, y, z, fmaxf(radius, 1.0f), fx, fy, fz, desc_.nx, desc_.ny, desc_.nz);
        check_cuda(cudaGetLastError(), "splat_force_compact_kernel launch");
    }

    void Solver3D::advect_scalar_(float* dst, const float* src, const float* u, const float* v, const float* w, Stream stream) {
        const dim3 block      = make_block(desc_);
        const dim3 grid       = make_grid(desc_.nx, desc_.ny, desc_.nz, block);
        const float dt_over_h = desc_.dt / desc_.cell_size;
        advect_scalar_kernel<<<grid, block, 0, stream>>>(dst, src, u, v, w, dt_over_h, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
        check_cuda(cudaGetLastError(), "advect_scalar_kernel launch");
        set_boundary_(0, dst, stream);
    }

    void Solver3D::advect_velocity_(float* u_dst, float* v_dst, float* w_dst, const float* u_src, const float* v_src, const float* w_src, const float* u_vel, const float* v_vel, const float* w_vel, Stream stream) {
        const dim3 block      = make_block(desc_);
        const dim3 grid       = make_grid(desc_.nx, desc_.ny, desc_.nz, block);
        const float dt_over_h = desc_.dt / desc_.cell_size;
        advect_velocity_kernel<<<grid, block, 0, stream>>>(u_dst, v_dst, w_dst, u_src, v_src, w_src, u_vel, v_vel, w_vel, dt_over_h, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
        check_cuda(cudaGetLastError(), "advect_velocity_kernel launch");
        set_boundary_(1, u_dst, stream);
        set_boundary_(2, v_dst, stream);
        set_boundary_(3, w_dst, stream);
    }

    void Solver3D::diffuse_scalar_(float* dst, const float* src, Stream stream) {
        if (desc_.diffusion <= 0.0f) {
            check_cuda(cudaMemcpyAsync(dst, src, ghosted_bytes_, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync scalar diffuse bypass");
            set_boundary_(0, dst, stream);
            return;
        }

        check_cuda(cudaMemcpyAsync(dst, src, ghosted_bytes_, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync scalar diffuse init");

        const dim3 block  = make_block(desc_);
        const dim3 grid   = make_grid(desc_.nx, desc_.ny, desc_.nz, block);
        const float alpha = desc_.dt * desc_.diffusion / (desc_.cell_size * desc_.cell_size);
        const float denom = 1.0f + 6.0f * alpha;

        set_boundary_(0, dst, stream);
        for (int iteration = 0; iteration < desc_.diffuse_iterations; ++iteration) {
            rbgs_diffuse_kernel<<<grid, block, 0, stream>>>(dst, src, alpha, denom, 0, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
            check_cuda(cudaGetLastError(), "rbgs_diffuse_kernel scalar red launch");
            set_boundary_(0, dst, stream);

            rbgs_diffuse_kernel<<<grid, block, 0, stream>>>(dst, src, alpha, denom, 1, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
            check_cuda(cudaGetLastError(), "rbgs_diffuse_kernel scalar black launch");
            set_boundary_(0, dst, stream);
        }
    }

    void Solver3D::diffuse_velocity_(float* u_dst, float* v_dst, float* w_dst, const float* u_src, const float* v_src, const float* w_src, Stream stream) {
        if (desc_.viscosity <= 0.0f) {
            check_cuda(cudaMemcpyAsync(u_dst, u_src, ghosted_bytes_, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync u diffuse bypass");
            check_cuda(cudaMemcpyAsync(v_dst, v_src, ghosted_bytes_, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync v diffuse bypass");
            check_cuda(cudaMemcpyAsync(w_dst, w_src, ghosted_bytes_, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync w diffuse bypass");
            set_boundary_(1, u_dst, stream);
            set_boundary_(2, v_dst, stream);
            set_boundary_(3, w_dst, stream);
            return;
        }

        check_cuda(cudaMemcpyAsync(u_dst, u_src, ghosted_bytes_, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync u diffuse init");
        check_cuda(cudaMemcpyAsync(v_dst, v_src, ghosted_bytes_, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync v diffuse init");
        check_cuda(cudaMemcpyAsync(w_dst, w_src, ghosted_bytes_, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync w diffuse init");

        const dim3 block  = make_block(desc_);
        const dim3 grid   = make_grid(desc_.nx, desc_.ny, desc_.nz, block);
        const float alpha = desc_.dt * desc_.viscosity / (desc_.cell_size * desc_.cell_size);
        const float denom = 1.0f + 6.0f * alpha;

        set_boundary_(1, u_dst, stream);
        set_boundary_(2, v_dst, stream);
        set_boundary_(3, w_dst, stream);

        for (int iteration = 0; iteration < desc_.diffuse_iterations; ++iteration) {
            rbgs_diffuse_kernel<<<grid, block, 0, stream>>>(u_dst, u_src, alpha, denom, 0, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
            rbgs_diffuse_kernel<<<grid, block, 0, stream>>>(v_dst, v_src, alpha, denom, 0, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
            rbgs_diffuse_kernel<<<grid, block, 0, stream>>>(w_dst, w_src, alpha, denom, 0, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
            check_cuda(cudaGetLastError(), "rbgs_diffuse_kernel velocity red launch");
            set_boundary_(1, u_dst, stream);
            set_boundary_(2, v_dst, stream);
            set_boundary_(3, w_dst, stream);

            rbgs_diffuse_kernel<<<grid, block, 0, stream>>>(u_dst, u_src, alpha, denom, 1, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
            rbgs_diffuse_kernel<<<grid, block, 0, stream>>>(v_dst, v_src, alpha, denom, 1, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
            rbgs_diffuse_kernel<<<grid, block, 0, stream>>>(w_dst, w_src, alpha, denom, 1, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
            check_cuda(cudaGetLastError(), "rbgs_diffuse_kernel velocity black launch");
            set_boundary_(1, u_dst, stream);
            set_boundary_(2, v_dst, stream);
            set_boundary_(3, w_dst, stream);
        }
    }

    void Solver3D::project_(float* u, float* v, float* w, Stream stream) {
        nvtx3::scoped_range range{"stable.step.project"};
        const dim3 block       = make_block(desc_);
        const dim3 grid        = make_grid(desc_.nx, desc_.ny, desc_.nz, block);
        const float half_inv_h = 0.5f / desc_.cell_size;
        const float h2         = desc_.cell_size * desc_.cell_size;

        divergence_kernel<<<grid, block, 0, stream>>>(divergence_, u, v, w, half_inv_h, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
        check_cuda(cudaGetLastError(), "divergence_kernel launch");
        set_boundary_(0, divergence_, stream);

        fill_field_(pressure_, 0.0f, stream);
        set_boundary_(0, pressure_, stream);

        for (int iteration = 0; iteration < desc_.pressure_iterations; ++iteration) {
            rbgs_pressure_kernel<<<grid, block, 0, stream>>>(pressure_, divergence_, h2, 0, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
            check_cuda(cudaGetLastError(), "rbgs_pressure_kernel red launch");
            set_boundary_(0, pressure_, stream);

            rbgs_pressure_kernel<<<grid, block, 0, stream>>>(pressure_, divergence_, h2, 1, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
            check_cuda(cudaGetLastError(), "rbgs_pressure_kernel black launch");
            set_boundary_(0, pressure_, stream);
        }

        set_boundary_(0, pressure_, stream);
        subtract_gradient_kernel<<<grid, block, 0, stream>>>(u, v, w, pressure_, half_inv_h, desc_.nx, desc_.ny, desc_.nz, nx_total_, ny_total_);
        check_cuda(cudaGetLastError(), "subtract_gradient_kernel launch");
        set_boundary_(1, u, stream);
        set_boundary_(2, v, stream);
        set_boundary_(3, w, stream);
    }

    void Solver3D::step(const FieldSet& fields, Stream stream) {
        nvtx3::scoped_range step_range{"stable.step"};
        validate_fields_(fields);

        copy_compact_to_ghosted_(density_, reinterpret_cast<const float*>(fields.density.values.data), stream);
        copy_compact_to_ghosted_(u_, reinterpret_cast<const float*>(fields.velocity.x.data), stream);
        copy_compact_to_ghosted_(v_, reinterpret_cast<const float*>(fields.velocity.y.data), stream);
        copy_compact_to_ghosted_(w_, reinterpret_cast<const float*>(fields.velocity.z.data), stream);

        set_boundary_(0, density_, stream);
        set_boundary_(1, u_, stream);
        set_boundary_(2, v_, stream);
        set_boundary_(3, w_, stream);

        {
            nvtx3::scoped_range range{"stable.step.advect_velocity"};
            advect_velocity_(u_prev_, v_prev_, w_prev_, u_, v_, w_, u_, v_, w_, stream);
        }
        {
            nvtx3::scoped_range range{"stable.step.diffuse_velocity"};
            diffuse_velocity_(u_, v_, w_, u_prev_, v_prev_, w_prev_, stream);
        }
        project_(u_, v_, w_, stream);

        {
            nvtx3::scoped_range range{"stable.step.advect_density"};
            advect_scalar_(density_prev_, density_, u_, v_, w_, stream);
        }
        {
            nvtx3::scoped_range range{"stable.step.diffuse_density"};
            diffuse_scalar_(density_, density_prev_, stream);
        }

        copy_ghosted_to_compact_(reinterpret_cast<float*>(fields.density.values.data), density_, stream);
        copy_ghosted_to_compact_(reinterpret_cast<float*>(fields.velocity.x.data), u_, stream);
        copy_ghosted_to_compact_(reinterpret_cast<float*>(fields.velocity.y.data), v_, stream);
        copy_ghosted_to_compact_(reinterpret_cast<float*>(fields.velocity.z.data), w_, stream);
    }

    void Solver3D::snapshot_density(const FieldSet& fields, const ScalarFieldT& destination, Stream stream) {
        nvtx3::scoped_range range{"stable.snapshot_density"};
        validate_fields_(fields);
        if (destination.grid.nx != desc_.nx || destination.grid.ny != desc_.ny || destination.grid.nz != desc_.nz) {
            throw std::invalid_argument("snapshot density destination grid dimensions must match the context");
        }
        if (std::fabs(destination.grid.cell_size - desc_.cell_size) > 1.0e-6f) {
            throw std::invalid_argument("snapshot density destination cell_size must match the context");
        }
        if (destination.values.data == nullptr) {
            throw std::invalid_argument("snapshot density destination must provide a non-null device pointer");
        }
        if (destination.values.format != FIELD_FORMAT_F32) {
            throw std::invalid_argument("snapshot density destination must use FIELD_FORMAT_F32");
        }
        if (destination.values.memory_type != FIELD_MEMORY_TYPE_CUDA_DEVICE) {
            throw std::invalid_argument("snapshot density destination must use FIELD_MEMORY_TYPE_CUDA_DEVICE");
        }
        if (destination.values.size_bytes < static_cast<std::uint64_t>(compact_bytes_)) {
            throw std::invalid_argument("snapshot density destination size_bytes is smaller than nx*ny*nz*sizeof(float)");
        }

        check_cuda(
            cudaMemcpyAsync(destination.values.data, fields.density.values.data, compact_bytes_, cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync density snapshot");
    }

    void Solver3D::snapshot_velocity_magnitude(const FieldSet& fields, const ScalarFieldT& destination, Stream stream) {
        nvtx3::scoped_range range{"stable.snapshot_velocity_magnitude"};
        validate_fields_(fields);
        if (destination.grid.nx != desc_.nx || destination.grid.ny != desc_.ny || destination.grid.nz != desc_.nz) {
            throw std::invalid_argument("snapshot velocity magnitude destination grid dimensions must match the context");
        }
        if (std::fabs(destination.grid.cell_size - desc_.cell_size) > 1.0e-6f) {
            throw std::invalid_argument("snapshot velocity magnitude destination cell_size must match the context");
        }
        if (destination.values.data == nullptr) {
            throw std::invalid_argument("snapshot velocity magnitude destination must provide a non-null device pointer");
        }
        if (destination.values.format != FIELD_FORMAT_F32) {
            throw std::invalid_argument("snapshot velocity magnitude destination must use FIELD_FORMAT_F32");
        }
        if (destination.values.memory_type != FIELD_MEMORY_TYPE_CUDA_DEVICE) {
            throw std::invalid_argument("snapshot velocity magnitude destination must use FIELD_MEMORY_TYPE_CUDA_DEVICE");
        }
        if (destination.values.size_bytes < static_cast<std::uint64_t>(compact_bytes_)) {
            throw std::invalid_argument("snapshot velocity magnitude destination size_bytes is smaller than nx*ny*nz*sizeof(float)");
        }

        constexpr int block_size = 256;
        const int grid_size = static_cast<int>((interior_cell_count_ + block_size - 1) / block_size);
        velocity_magnitude_kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<float*>(destination.values.data),
            reinterpret_cast<const float*>(fields.velocity.x.data),
            reinterpret_cast<const float*>(fields.velocity.y.data),
            reinterpret_cast<const float*>(fields.velocity.z.data),
            interior_cell_count_);
        check_cuda(cudaGetLastError(), "velocity_magnitude_kernel launch");
    }

} // namespace stable_fluids

struct StableFluidsContext {
    stable_fluids::Solver3D* solver;
    std::string last_error;
};

namespace {

    [[nodiscard]] stable_fluids::Stream to_stream(void* cuda_stream) noexcept {
        return reinterpret_cast<stable_fluids::Stream>(cuda_stream);
    }

    void set_global_error(const char* message) {
        stable_fluids::g_last_error = message != nullptr ? message : "unknown stable-fluids error";
    }

    int32_t store_error(StableFluidsContext* context, int32_t code, const char* message) {
        set_global_error(message);
        if (context != nullptr) {
            context->last_error = stable_fluids::g_last_error;
        }
        return code;
    }

    int32_t copy_error_text(const std::string& text, char* buffer, std::uint64_t buffer_size) {
        if (buffer == nullptr || buffer_size == 0) {
            return STABLE_FLUIDS_ERROR_BUFFER_TOO_SMALL;
        }
        if (buffer_size <= static_cast<std::uint64_t>(text.size())) {
            buffer[0] = '\0';
            return STABLE_FLUIDS_ERROR_BUFFER_TOO_SMALL;
        }

        std::memcpy(buffer, text.c_str(), text.size() + 1);
        return STABLE_FLUIDS_SUCCESS;
    }

    template <class Fn>
    int32_t stable_fluids_try(StableFluidsContext* context, Fn&& fn) {
        try {
            fn();
            stable_fluids::g_last_error.clear();
            if (context != nullptr) {
                context->last_error.clear();
            }
            return STABLE_FLUIDS_SUCCESS;
        } catch (const std::bad_alloc& ex) {
            return store_error(context, STABLE_FLUIDS_ERROR_ALLOCATION_FAILED, ex.what());
        } catch (const std::invalid_argument& ex) {
            return store_error(context, STABLE_FLUIDS_ERROR_INVALID_ARGUMENT, ex.what());
        } catch (const std::exception& ex) {
            return store_error(context, STABLE_FLUIDS_ERROR_RUNTIME, ex.what());
        } catch (...) {
            return store_error(context, STABLE_FLUIDS_ERROR_RUNTIME, "unknown stable-fluids exception");
        }
    }

} // namespace

extern "C" {

StableFluidsContextDesc stable_fluids_context_desc_default(void) {
    return StableFluidsContextDesc{96, 96, 96, 1.0f / 60.0f, 1.0f, 0.0001f, 0.00005f, 20, 80, 8, 8, 8};
}

StableFluidsContext* stable_fluids_context_create(const StableFluidsContextDesc* desc) {
    if (desc == nullptr) {
        set_global_error("stable_fluids_context_create received a null desc");
        return nullptr;
    }

    try {
        auto* context = new StableFluidsContext{new stable_fluids::Solver3D(*desc), {}};
        stable_fluids::g_last_error.clear();
        return context;
    } catch (const std::exception& ex) {
        set_global_error(ex.what());
        return nullptr;
    } catch (...) {
        set_global_error("unknown stable-fluids exception");
        return nullptr;
    }
}

void stable_fluids_context_destroy(StableFluidsContext* context) {
    if (context == nullptr) {
        return;
    }
    delete context->solver;
    delete context;
}

uint64_t stable_fluids_context_required_elements(const StableFluidsContext* context) {
    return context != nullptr && context->solver != nullptr ? context->solver->required_elements() : 0;
}

uint64_t stable_fluids_context_required_scalar_field_bytes(const StableFluidsContext* context) {
    return context != nullptr && context->solver != nullptr ? context->solver->required_scalar_field_bytes() : 0;
}

uint64_t stable_fluids_context_required_vector_field_component_bytes(const StableFluidsContext* context, uint32_t component) {
    return context != nullptr && context->solver != nullptr ? context->solver->required_vector_field_component_bytes(component) : 0;
}

int32_t stable_fluids_fields_clear_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr || fields == nullptr) {
        return store_error(context, STABLE_FLUIDS_ERROR_INVALID_ARGUMENT, "stable_fluids_fields_clear_async received a null argument");
    }
    return stable_fluids_try(context, [&] { context->solver->zero_fields(*fields, to_stream(cuda_stream)); });
}

int32_t stable_fluids_fields_step_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr || fields == nullptr) {
        return store_error(context, STABLE_FLUIDS_ERROR_INVALID_ARGUMENT, "stable_fluids_fields_step_async received a null argument");
    }
    return stable_fluids_try(context, [&] { context->solver->step(*fields, to_stream(cuda_stream)); });
}

int32_t stable_fluids_fields_add_density_splat_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, const StableFluidsDensitySplatDesc* splat, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr || fields == nullptr || splat == nullptr) {
        return store_error(context, STABLE_FLUIDS_ERROR_INVALID_ARGUMENT, "stable_fluids_fields_add_density_splat_async received a null argument");
    }
    return stable_fluids_try(context, [&] { context->solver->add_density_sphere(*fields, splat->center_x, splat->center_y, splat->center_z, splat->radius, splat->amount, to_stream(cuda_stream)); });
}

int32_t stable_fluids_fields_add_force_splat_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, const StableFluidsForceSplatDesc* splat, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr || fields == nullptr || splat == nullptr) {
        return store_error(context, STABLE_FLUIDS_ERROR_INVALID_ARGUMENT, "stable_fluids_fields_add_force_splat_async received a null argument");
    }
    return stable_fluids_try(context, [&] { context->solver->add_force_sphere(*fields, splat->center_x, splat->center_y, splat->center_z, splat->radius, splat->force_x, splat->force_y, splat->force_z, to_stream(cuda_stream)); });
}

int32_t stable_fluids_fields_snapshot_density_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, const ScalarField* destination, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr || fields == nullptr || destination == nullptr) {
        return store_error(context, STABLE_FLUIDS_ERROR_INVALID_ARGUMENT, "stable_fluids_fields_snapshot_density_async received a null argument");
    }
    return stable_fluids_try(context, [&] { context->solver->snapshot_density(*fields, *destination, to_stream(cuda_stream)); });
}

int32_t stable_fluids_fields_snapshot_velocity_magnitude_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, const ScalarField* destination, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr || fields == nullptr || destination == nullptr) {
        return store_error(context, STABLE_FLUIDS_ERROR_INVALID_ARGUMENT, "stable_fluids_fields_snapshot_velocity_magnitude_async received a null argument");
    }
    return stable_fluids_try(context, [&] { context->solver->snapshot_velocity_magnitude(*fields, *destination, to_stream(cuda_stream)); });
}

uint64_t stable_fluids_last_error_length(void) {
    return static_cast<uint64_t>(stable_fluids::g_last_error.size());
}

uint64_t stable_fluids_context_last_error_length(const StableFluidsContext* context) {
    return context != nullptr ? static_cast<uint64_t>(context->last_error.size()) : static_cast<uint64_t>(stable_fluids::g_last_error.size());
}

int32_t stable_fluids_copy_last_error(char* buffer, uint64_t buffer_size) {
    return copy_error_text(stable_fluids::g_last_error, buffer, buffer_size);
}

int32_t stable_fluids_copy_context_last_error(const StableFluidsContext* context, char* buffer, uint64_t buffer_size) {
    return copy_error_text(context != nullptr ? context->last_error : stable_fluids::g_last_error, buffer, buffer_size);
}

} // extern "C"
