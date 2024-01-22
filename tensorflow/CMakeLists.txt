add_executable(tensorflow-load load.cc)
target_link_libraries(tensorflow-load PUBLIC LibTensorflow Eigen3::Eigen)