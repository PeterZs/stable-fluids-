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

#ifdef __cplusplus
extern "C" {
#endif

/*
Error code scheme:
0     : success
1xxx  : scalar/grid/step parameter errors
1001  : invalid grid dimensions
1002  : invalid cell size
1003  : invalid dt
1004  : invalid iteration count
1005  : invalid source radius
2xxx  : buffer errors
2001  : invalid density buffer
2003  : invalid velocity_x buffer
2004  : invalid velocity_y buffer
2005  : invalid velocity_z buffer
2006  : invalid destination buffer
2007  : invalid temporary density buffer
2008  : invalid temporary velocity_x buffer
2009  : invalid temporary velocity_y buffer
2010  : invalid temporary velocity_z buffer
2011  : invalid temporary previous density buffer
2012  : invalid temporary previous velocity_x buffer
2013  : invalid temporary previous velocity_y buffer
2014  : invalid temporary previous velocity_z buffer
2015  : invalid temporary pressure buffer
2016  : invalid temporary divergence buffer
5xxx  : CUDA runtime or kernel launch failure
5001  : CUDA call failed
*/

STABLE_FLUIDS_API uint64_t stable_fluids_scalar_field_bytes(int32_t nx, int32_t ny, int32_t nz);
STABLE_FLUIDS_API uint64_t stable_fluids_temporary_field_bytes(int32_t nx, int32_t ny, int32_t nz);

STABLE_FLUIDS_API int32_t stable_fluids_clear_async(void* density, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, void* cuda_stream);

STABLE_FLUIDS_API int32_t stable_fluids_add_density_splat_async(void* density, int32_t nx, int32_t ny, int32_t nz, float center_x, float center_y, float center_z, float radius, float amount, void* cuda_stream);

STABLE_FLUIDS_API int32_t stable_fluids_add_force_splat_async(void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, float center_x, float center_y, float center_z, float radius, float force_x, float force_y, float force_z, void* cuda_stream);

STABLE_FLUIDS_API int32_t stable_fluids_step_async(void* density, void* velocity_x, void* velocity_y, void* velocity_z, int32_t nx, int32_t ny, int32_t nz, float cell_size, void* temporary_density, void* temporary_velocity_x, void* temporary_velocity_y, void* temporary_velocity_z, void* temporary_previous_density, void* temporary_previous_velocity_x,
    void* temporary_previous_velocity_y, void* temporary_previous_velocity_z, void* temporary_pressure, void* temporary_divergence, float dt, float viscosity, float diffusion, int32_t diffuse_iterations, int32_t pressure_iterations, int32_t block_x, int32_t block_y, int32_t block_z, void* cuda_stream);

STABLE_FLUIDS_API int32_t stable_fluids_compute_velocity_magnitude_async(void* velocity_x, void* velocity_y, void* velocity_z, void* destination, int32_t nx, int32_t ny, int32_t nz, void* cuda_stream);

#ifdef __cplusplus
}
#endif

#endif // STABLE_FLUIDS_H
