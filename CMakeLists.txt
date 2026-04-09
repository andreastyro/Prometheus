cmake_minimum_required(VERSION 3.10)
project(prometheus)

set(CMAKE_CXX_STANDARD 17)

# Collect all source files
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loss.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dataloader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/linear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/activations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/dropout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/batchnorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/sequential.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/flatten.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/rnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/lstm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/gru.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/transformer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/embedding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optim/sgd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optim/adam.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optim/rmsprop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metrics/classification.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metrics/regression.cpp
)

# Build the library
add_library(prometheus ${SOURCES})
target_include_directories(prometheus PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build the test executable
add_executable(test_tensor ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor.cpp)
target_link_libraries(test_tensor prometheus)
