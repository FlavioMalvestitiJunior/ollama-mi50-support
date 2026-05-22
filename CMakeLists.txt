cmake_minimum_required(VERSION 3.21)
project(ollama-ggml-hip LANGUAGES C CXX)

set(GGML_SRC /home/llm/ollama-src/ml/backend/ggml/ggml)
set(ROCM_PATH /opt/rocm)

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

# Set gfx906 target
set(CMAKE_HIP_ARCHITECTURES "gfx906")
enable_language(HIP)

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

# Collect sources matching ggml-hip/CMakeLists.txt
file(GLOB   GGML_SOURCES_ROCM "${GGML_SRC}/src/ggml-cuda/*.cu")
file(GLOB   SRCS "${GGML_SRC}/src/ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "${GGML_SRC}/src/ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "${GGML_SRC}/src/ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "${GGML_SRC}/src/ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "${GGML_SRC}/src/ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "${GGML_SRC}/src/ggml-cuda/template-instances/fattn-vec*f16-f16.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

add_library(ggml-hip MODULE ${GGML_SOURCES_ROCM})

set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --gpu-max-threads-per-block=1024")

target_compile_definitions(ggml-hip PRIVATE
    GGML_USE_HIP
    GGML_USE_CUDA
    GGML_BACKEND_DL
    NDEBUG
    GGML_CUDA_FORCE_MMQ
    GGML_HIP_NO_VMM
)

target_include_directories(ggml-hip PRIVATE
    "${GGML_SRC}/include"
    "${GGML_SRC}/src"
    "${GGML_SRC}/src/ggml-cuda"
)

target_link_libraries(ggml-hip PRIVATE
    /usr/local/lib/ollama/libggml-base.so
    hip::host
    roc::rocblas
    roc::hipblas
)

set_target_properties(ggml-hip PROPERTIES
    OUTPUT_NAME "ggml-hip"
    PREFIX "lib"
    SUFFIX ".so"
)
