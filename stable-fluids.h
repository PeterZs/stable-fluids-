#ifndef STABLE_FLUIDS_H
#define STABLE_FLUIDS_H

#include <stdint.h>

#ifndef SMOKE_FIELD_TYPES_DEFINED
#define SMOKE_FIELD_TYPES_DEFINED

#define FIELD_FORMAT_F32 1u
#define FIELD_MEMORY_TYPE_CUDA_DEVICE 1u

typedef struct FieldGridDesc {
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
} FieldGridDesc;

typedef struct FieldBufferView {
    void* data;
    uint64_t size_bytes;
    uint32_t format;
    uint32_t memory_type;
} FieldBufferView;

typedef struct ScalarField {
    FieldGridDesc grid;
    FieldBufferView values;
} ScalarField;

typedef enum VectorFieldLayout {
    VECTOR_FIELD_LAYOUT_CELL_CENTERED = 1,
    VECTOR_FIELD_LAYOUT_STAGGERED_MAC = 2,
} VectorFieldLayout;

typedef enum VectorFieldComponent {
    VECTOR_FIELD_COMPONENT_X = 0,
    VECTOR_FIELD_COMPONENT_Y = 1,
    VECTOR_FIELD_COMPONENT_Z = 2,
} VectorFieldComponent;

typedef struct VectorField {
    FieldGridDesc grid;
    uint32_t layout;
    FieldBufferView x;
    FieldBufferView y;
    FieldBufferView z;
} VectorField;

#endif

#if defined(_WIN32)
#if defined(STABLE_FLUIDS_BUILD_SHARED)
#define STABLE_FLUIDS_API __declspec(dllexport)
#else
#define STABLE_FLUIDS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define STABLE_FLUIDS_API __attribute__((visibility("default")))
#else
#define STABLE_FLUIDS_API
#endif

#define STABLE_FLUIDS_SUCCESS                 0
#define STABLE_FLUIDS_ERROR_INVALID_ARGUMENT  -1
#define STABLE_FLUIDS_ERROR_RUNTIME           -2
#define STABLE_FLUIDS_ERROR_ALLOCATION_FAILED -3
#define STABLE_FLUIDS_ERROR_BUFFER_TOO_SMALL  -4

#ifdef __cplusplus
extern "C" {
#endif

typedef struct StableFluidsContextDesc {
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float dt;
    float cell_size;
    float viscosity;
    float diffusion;
    int32_t diffuse_iterations;
    int32_t pressure_iterations;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
} StableFluidsContextDesc;

typedef struct StableFluidsFieldSet {
    ScalarField density;
    VectorField velocity;
} StableFluidsFieldSet;

typedef struct StableFluidsDensitySplatDesc {
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float amount;
} StableFluidsDensitySplatDesc;

typedef struct StableFluidsForceSplatDesc {
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float force_x;
    float force_y;
    float force_z;
} StableFluidsForceSplatDesc;

typedef struct StableFluidsContext StableFluidsContext;

STABLE_FLUIDS_API StableFluidsContextDesc stable_fluids_context_desc_default(void);
STABLE_FLUIDS_API StableFluidsContext* stable_fluids_context_create(const StableFluidsContextDesc* desc);
STABLE_FLUIDS_API void stable_fluids_context_destroy(StableFluidsContext* context);

STABLE_FLUIDS_API uint64_t stable_fluids_context_required_elements(const StableFluidsContext* context);
STABLE_FLUIDS_API uint64_t stable_fluids_context_required_scalar_field_bytes(const StableFluidsContext* context);
STABLE_FLUIDS_API uint64_t stable_fluids_context_required_vector_field_component_bytes(const StableFluidsContext* context, uint32_t component);

STABLE_FLUIDS_API int32_t stable_fluids_fields_clear_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, void* cuda_stream);
STABLE_FLUIDS_API int32_t stable_fluids_fields_step_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, void* cuda_stream);
STABLE_FLUIDS_API int32_t stable_fluids_fields_add_density_splat_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, const StableFluidsDensitySplatDesc* splat, void* cuda_stream);
STABLE_FLUIDS_API int32_t stable_fluids_fields_add_force_splat_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, const StableFluidsForceSplatDesc* splat, void* cuda_stream);
STABLE_FLUIDS_API int32_t stable_fluids_fields_snapshot_density_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, const ScalarField* destination, void* cuda_stream);
STABLE_FLUIDS_API int32_t stable_fluids_fields_snapshot_velocity_magnitude_async(StableFluidsContext* context, const StableFluidsFieldSet* fields, const ScalarField* destination, void* cuda_stream);

STABLE_FLUIDS_API uint64_t stable_fluids_last_error_length(void);
STABLE_FLUIDS_API uint64_t stable_fluids_context_last_error_length(const StableFluidsContext* context);
STABLE_FLUIDS_API int32_t stable_fluids_copy_last_error(char* buffer, uint64_t buffer_size);
STABLE_FLUIDS_API int32_t stable_fluids_copy_context_last_error(const StableFluidsContext* context, char* buffer, uint64_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif // STABLE_FLUIDS_H
