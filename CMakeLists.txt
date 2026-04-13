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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dataloader.cpp
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
