cmake_minimum_required(VERSION 3.15)
set(CMAKE_TOOLCHAIN_FILE
    "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    CACHE FILEPATH "vcpkg toolchain file")

project(camcal_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Using ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    set(ENV{CCACHE_DEBUG} "1")

endif(CCACHE_FOUND)

if(APPLE)
    set(INSTALL_PATH "@loader_path")
else()
    set(INSTALL_PATH "$ORIGIN")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)


pybind11_add_module(camcal_bindings MODULE cpp_src/main.cpp)

install(TARGETS camcal_bindings DESTINATION camcal)