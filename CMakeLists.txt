cmake_minimum_required(VERSION 3.10)
project(prometheus)

set(CMAKE_CXX_STANDARD 17)

# Collect all source files
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loss.cpp
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
)

# Build the library
add_library(prometheus ${SOURCES})
target_include_directories(prometheus PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build the test executables
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
