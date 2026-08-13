cmake_minimum_required(VERSION 3.10)
project(prometheus)

set(CMAKE_CXX_STANDARD 17)

# ── Backend options ────────────────────────────────────────────────────────────
# Pass -DUSE_OPENBLAS=ON, -DUSE_MKL=ON, or -DUSE_OPENMP=ON to cmake to enable.
# Priority: OpenBLAS > MKL > OpenMP > plain tiled matmul (default fallback).
option(USE_OPENBLAS "Use OpenBLAS for matmul (best on AMD)"   OFF)
option(USE_MKL      "Use Intel MKL for matmul (best on Intel)" OFF)
option(USE_OPENMP   "Parallelise matmul across CPU cores"      OFF)

# ── Source files ──────────────────────────────────────────────────────────────
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loss.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/matmul_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/linear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/activations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/dropout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/sequential.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optim/sgd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optim/adam.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optim/rmsprop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/dataloader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/csv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/model_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/trainer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/summary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/early_stopping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metrics/metrics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/flatten.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/vision/conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/vision/residual_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/vision/maxpool2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/vision/avgpool2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/vision/convtranspose2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/rnn/rnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/rnn/lstm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/rnn/gru.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optim/grad_scaler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/nlp/embedding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/nlp/layernorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/groupnorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/nlp/attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/nlp/positional_encoding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/nlp/transformer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optim/adamw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/grad_clip.cpp
)

# ── Build the library ─────────────────────────────────────────────────────────
add_library(prometheus ${SOURCES})
target_include_directories(prometheus PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Backend detection & linking ───────────────────────────────────────────────
if(USE_OPENBLAS)
    # If OPENBLAS_ROOT is set, use it directly (Windows pre-built zip)
    if(OPENBLAS_ROOT)
        message(STATUS "Backend: OpenBLAS (from OPENBLAS_ROOT=${OPENBLAS_ROOT})")
        target_compile_definitions(prometheus PUBLIC PROMETHEUS_USE_OPENBLAS)
        target_include_directories(prometheus PUBLIC "${OPENBLAS_ROOT}/include")
        target_link_libraries(prometheus "${OPENBLAS_ROOT}/lib/libopenblas.dll.a")
    else()
        # Linux/macOS: find via pkg-config or system paths
        find_library(OPENBLAS_LIB NAMES openblas)
        find_path(OPENBLAS_INCLUDE NAMES cblas.h PATH_SUFFIXES openblas)
        if(OPENBLAS_LIB AND OPENBLAS_INCLUDE)
            message(STATUS "Backend: OpenBLAS (system)")
            target_compile_definitions(prometheus PUBLIC PROMETHEUS_USE_OPENBLAS)
            target_include_directories(prometheus PUBLIC ${OPENBLAS_INCLUDE})
            target_link_libraries(prometheus ${OPENBLAS_LIB})
        else()
            message(WARNING "OpenBLAS not found — falling back to tiled matmul")
        endif()
    endif()

elseif(USE_MKL)
    find_package(MKL QUIET)
    if(MKL_FOUND)
        message(STATUS "Backend: Intel MKL ${MKL_VERSION}")
        target_compile_definitions(prometheus PUBLIC PROMETHEUS_USE_MKL)
        target_include_directories(prometheus PUBLIC ${MKL_INCLUDE_DIRS})
        target_link_libraries(prometheus ${MKL_LIBRARIES})
    else()
        message(WARNING "MKL not found — falling back to tiled matmul")
    endif()

else()
    message(STATUS "Backend: tiled cache-friendly matmul (no external BLAS)")
endif()

if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP: enabled (${OpenMP_CXX_VERSION})")
        target_compile_definitions(prometheus PUBLIC PROMETHEUS_USE_OPENMP)
        target_link_libraries(prometheus OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found — running single-threaded")
    endif()
endif()

# ── Test executables ──────────────────────────────────────────────────────────
add_executable(test_tensor ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor.cpp)
target_link_libraries(test_tensor prometheus)

add_executable(test_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model.cpp)
target_link_libraries(test_model prometheus)

add_executable(test_dataloader ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dataloader.cpp)
target_link_libraries(test_dataloader prometheus)

add_executable(test_csv ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_csv.cpp)
target_link_libraries(test_csv prometheus)

add_executable(test_image ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_image.cpp)
target_link_libraries(test_image prometheus)

add_executable(test_model_io ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model_io.cpp)
target_link_libraries(test_model_io prometheus)

add_executable(test_trainer ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_trainer.cpp)
target_link_libraries(test_trainer prometheus)

add_executable(test_summary ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_summary.cpp)
target_link_libraries(test_summary prometheus)

add_executable(test_early_stopping ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_early_stopping.cpp)
target_link_libraries(test_early_stopping prometheus)

add_executable(test_metrics ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_metrics.cpp)
target_link_libraries(test_metrics prometheus)

add_executable(test_conv2d ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conv2d.cpp)
target_link_libraries(test_conv2d prometheus)

add_executable(test_maxpool2d ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_maxpool2d.cpp)
target_link_libraries(test_maxpool2d prometheus)

add_executable(test_convtranspose2d ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_convtranspose2d.cpp)
target_link_libraries(test_convtranspose2d prometheus)

add_executable(test_rnn ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rnn.cpp)
target_link_libraries(test_rnn prometheus)

add_executable(test_lstm ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lstm.cpp)
target_link_libraries(test_lstm prometheus)

add_executable(test_gru ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gru.cpp)
target_link_libraries(test_gru prometheus)

add_executable(test_embedding ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_embedding.cpp)
target_link_libraries(test_embedding prometheus)

add_executable(test_layernorm ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_layernorm.cpp)
target_link_libraries(test_layernorm prometheus)

add_executable(test_groupnorm ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_groupnorm.cpp)
target_link_libraries(test_groupnorm prometheus)

add_executable(test_attention ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_attention.cpp)
target_link_libraries(test_attention prometheus)

add_executable(test_transformer ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_transformer.cpp)
target_link_libraries(test_transformer prometheus)

add_executable(benchmark_matmul ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_matmul.cpp)
target_link_libraries(benchmark_matmul prometheus)

add_executable(test_grad_clip   ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_grad_clip.cpp)
target_link_libraries(test_grad_clip prometheus)

add_executable(test_adamw       ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_adamw.cpp)
target_link_libraries(test_adamw prometheus)

add_executable(test_checkpoint  ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_checkpoint.cpp)
target_link_libraries(test_checkpoint prometheus)
