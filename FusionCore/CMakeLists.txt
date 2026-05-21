cmake_minimum_required(VERSION 3.18)
project(FusionCore)

set(CMAKE_CXX_STANDARD 17)

# ----- Фиксация Python из окружения проекта (untitled4) -----
# set(Python3_ROOT "C:/Users/Фёдор/.conda/envs/untitled4")  # закомментировать для ноута
# set(Python3_FIND_STRATEGY LOCATION)  # закомментировать для ноута
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")

# ----- Указываем путь к pybind11 вручную (замени на свой) -----
set(pybind11_DIR "C:/Users/Фёдор/.conda/envs/untitled4/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Создание модуль
pybind11_add_module(_fusion_core library.cpp)
target_include_directories(_fusion_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})