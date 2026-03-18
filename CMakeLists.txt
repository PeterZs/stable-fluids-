cmake_minimum_required(VERSION 4.1.2)

project(stable-fluids VERSION 0.3.0 LANGUAGES CXX CUDA)

# ============================================================================
# Toolchain
# ============================================================================
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES native)

# ============================================================================
# Targets
# ============================================================================
add_library(
        stable-fluids
        SHARED
        src/stable_fluids.h
        src/solver.cu
)

add_executable(
        stable-fluids-app
        main.cpp
)

target_compile_features(stable-fluids PRIVATE cxx_std_23)
target_compile_features(stable-fluids-app PRIVATE cxx_std_23)

target_compile_definitions(stable-fluids PRIVATE STABLE_FLUIDS_BUILD_SHARED)

target_include_directories(stable-fluids PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(stable-fluids-app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(stable-fluids PRIVATE CUDA::cudart)
target_link_libraries(stable-fluids-app PRIVATE stable-fluids CUDA::cudart)

# ============================================================================
# Compile Options
# ============================================================================
if (MSVC)
    target_compile_options(stable-fluids PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
    target_compile_options(stable-fluids-app PRIVATE /W4 /permissive-)
else ()
    target_compile_options(stable-fluids PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    target_compile_options(stable-fluids-app PRIVATE -Wall -Wextra -Wpedantic)
endif ()

target_compile_options(
        stable-fluids
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

# ============================================================================
# CUDA Linking
# ============================================================================
set_target_properties(
        stable-fluids
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ============================================================================
# Output Layout
# ============================================================================
foreach (target stable-fluids stable-fluids-app)
    set_target_properties(
            ${target}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach ()

# ============================================================================
# Runtime Lookup
# ============================================================================
if (UNIX)
    foreach (target stable-fluids stable-fluids-app)
        set_target_properties(
                ${target}
                PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
        )
    endforeach ()
endif ()
