cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_bart_matmul)

add_executable(metal_example_bart_matmul)
target_sources(metal_example_bart_matmul PRIVATE bart_matmul.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_bart_matmul PUBLIC TT::Metalium)
