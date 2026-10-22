cmake_minimum_required(VERSION 3.19)
project(edgeforge_qwen3_vl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Same BMRT/pybind11 interface as the pinned official Qwen3_VL python_demo.
find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
set(SOPHON_SDK_ROOT "/opt/sophon/libsophon-current" CACHE PATH "Installed libsophon development SDK")
find_path(SOPHON_INCLUDE bmruntime_interface.h PATHS "${SOPHON_SDK_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(BMRT_LIBRARY bmrt PATHS "${SOPHON_SDK_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(BMLIB_LIBRARY bmlib PATHS "${SOPHON_SDK_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
pybind11_add_module(edgeforge_qwen3_vl chat.cpp)
target_include_directories(edgeforge_qwen3_vl PRIVATE "${SOPHON_INCLUDE}")
target_link_libraries(edgeforge_qwen3_vl PRIVATE "${BMRT_LIBRARY}" "${BMLIB_LIBRARY}")
# Keep upstream runtime assertions enabled in Release builds too.
target_compile_options(edgeforge_qwen3_vl PRIVATE -UNDEBUG)
set_target_properties(edgeforge_qwen3_vl PROPERTIES INSTALL_RPATH "${SOPHON_SDK_ROOT}/lib")
execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
  OUTPUT_VARIABLE PYTHON_SITE OUTPUT_STRIP_TRAILING_WHITESPACE)
install(TARGETS edgeforge_qwen3_vl LIBRARY DESTINATION "${PYTHON_SITE}")
