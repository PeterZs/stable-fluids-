#ifndef STABLE_FLUIDS_H
#define STABLE_FLUIDS_H

#include <stdint.h>

#ifdef _WIN32
#ifdef STABLE_FLUIDS_BUILD_SHARED
#define STABLE_FLUIDS_API __declspec(dllexport)
#else
#define STABLE_FLUIDS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define STABLE_FLUIDS_API __attribute__((visibility("default")))
#else
#define STABLE_FLUIDS_API
#endif

#define STABLE_FLUIDS_SUCCESS 0

#ifdef __cplusplus
extern "C" {
#endif

STABLE_FLUIDS_API uint64_t stable_fluids_scalar_field_bytes(int32_t nx, int32_t ny, int32_t nz);
STABLE_FLUIDS_API uint64_t stable_fluids_workspace_bytes(int32_t nx, int32_t ny, int32_t nz);

STABLE_FLUIDS_API int32_t stable_fluids_clear_async(
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
    void* cuda_stream);

STABLE_FLUIDS_API int32_t stable_fluids_add_density_splat_async(
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
    void* cuda_stream);

STABLE_FLUIDS_API int32_t stable_fluids_add_force_splat_async(
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
    void* cuda_stream);

STABLE_FLUIDS_API int32_t stable_fluids_step_async(
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
    void* cuda_stream);

STABLE_FLUIDS_API int32_t stable_fluids_compute_velocity_magnitude_async(
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
    void* cuda_stream);

#ifdef __cplusplus
}
#endif

#endif // STABLE_FLUIDS_H
