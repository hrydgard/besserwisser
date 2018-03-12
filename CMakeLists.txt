CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

# Build speedup
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_STANDARD 14)

add_definitions(-mssse3)
add_definitions(-mavx)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O2")

project (besserwisser)
 
set(FILES
  layer.cpp
  layer.h
  main.cpp
  math_util.cpp
  math_util.h
  mnist.cpp
  mnist.h
  network.cpp
  network.h
  train.cpp
  train.h
  blob.cpp
  blob.h
)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

add_executable(besserwisser ${FILES})
