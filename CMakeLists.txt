cmake_minimum_required(VERSION 3.10)
project(GKD_Model)

set(CMAKE_CXX_STANDARD 17)

# 告诉 CMake 使用 OpenCV
find_package(OpenCV REQUIRED)

# 添加你的源码
add_executable(main code.cpp)

# 链接 OpenCV
target_link_libraries(main PRIVATE ${OpenCV_LIBS})

# 包含 OpenCV 的头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})
