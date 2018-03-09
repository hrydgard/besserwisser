CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

project (simplenet)
 
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
)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  
add_executable(simplenet ${FILES})