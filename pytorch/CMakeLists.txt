add_executable(pytorch-load load.cc)
target_link_libraries(pytorch-load PUBLIC PyTorch Eigen3::Eigen)