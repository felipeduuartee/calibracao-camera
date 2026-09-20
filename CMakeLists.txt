cmake_minimum_required(VERSION 3.16)

project(calibracao_camera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(calibra_camera
    src/main.cpp
    src/calibracao.cpp
)

target_include_directories(calibra_camera PRIVATE
    src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(calibra_camera PRIVATE
    ${OpenCV_LIBS}
)
