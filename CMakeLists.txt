cmake_minimum_required(VERSION 3.14)
project(microscope LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Apple frameworks
find_library(COREML_FRAMEWORK CoreML REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(AVFOUNDATION_FRAMEWORK AVFoundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
find_library(COREMEDIA_FRAMEWORK CoreMedia REQUIRED)
find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)

# Sources
set(SOURCES
    src/main.cpp
    src/tokenizer.cpp
    src/image_utils.cpp
    src/pipeline.cpp
    src/coreml_model.mm
    src/camera.mm
    src/display.mm
)

add_executable(microscope ${SOURCES})

target_include_directories(microscope PRIVATE
    src
    thirdparty
)

target_link_libraries(microscope PRIVATE
    ${COREML_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
    ${AVFOUNDATION_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    ${COCOA_FRAMEWORK}
    ${COREMEDIA_FRAMEWORK}
    ${COREVIDEO_FRAMEWORK}
)

# ARC for ObjC++ files
set_source_files_properties(
    src/coreml_model.mm
    src/camera.mm
    src/display.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

# -ffast-math for hot-path files (excludes tokenizer.cpp which includes json.hpp using infinity)
set_source_files_properties(
    src/main.cpp
    src/pipeline.cpp
    PROPERTIES COMPILE_FLAGS "-ffast-math"
)
set_source_files_properties(
    src/image_utils.cpp
    PROPERTIES COMPILE_FLAGS "-ffast-math -Wno-deprecated-declarations"
)

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(microscope PRIVATE
        -O3
        -mcpu=native
        -DNDEBUG
        -flto=auto
    )
    target_link_options(microscope PRIVATE -flto=auto)
endif()
