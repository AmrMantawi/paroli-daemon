cmake_minimum_required(VERSION 3.15)
project(paroli LANGUAGES CXX)

option(USE_RKNN "Enable RKNN for accelerated inference" OFF)
option(BUILD_DAEMON "Build paroli-daemon" ON)
# Server has been removed; only CLI and daemon remain

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

include_directories(${PIPER_PHONEMIZE_ROOT}/include)
link_directories(${PIPER_PHONEMIZE_ROOT}/lib)

# Add system include directories (only for native builds)
if(NOT CMAKE_CROSSCOMPILING)
    include_directories(SYSTEM /usr/include /usr/include/opus)
endif()

# find_package(onnxruntime REQUIRED)
# find_package(piper_phonemize REQUIRED)
find_package(fmt REQUIRED)
find_package(Opus REQUIRED)

# Find ALSA for audio playback
find_library(ALSA_LIBRARY asound)
if(NOT ALSA_LIBRARY)
    message(FATAL_ERROR "ALSA library not found. Install with: sudo apt install libasound2-dev")
endif()

add_library(piper
    piper/piper.cpp)

if (USE_RKNN)
    target_compile_definitions(piper PRIVATE USE_RKNN)
    target_sources(piper PRIVATE piper/rknn-inferer.cpp)
    target_link_libraries(piper PRIVATE rknnrt)
endif()
target_precompile_headers(piper PRIVATE piper/pch.hpp)

target_link_libraries(piper
    PRIVATE
    onnxruntime
    espeak-ng
    piper_phonemize
    pthread
    fmt::fmt
    dl)

target_include_directories(piper
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/piper)

add_executable(paroli-cli
    paroli-cli/main.cpp)
target_link_libraries(paroli-cli PRIVATE piper)


if (BUILD_DAEMON)
    add_library(paroli-daemon-lib
        paroli-daemon/paroli_daemon.cpp
        paroli-daemon/OggOpusEncoder.cpp)
    target_link_libraries(paroli-daemon-lib PRIVATE piper soxr ${OPUS_LIBRARIES} opusenc ogg ${ALSA_LIBRARY})
    target_include_directories(paroli-daemon-lib PRIVATE ${OPUS_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(paroli-daemon
        paroli-daemon/main.cpp)
    target_link_libraries(paroli-daemon PRIVATE paroli-daemon-lib)
    target_include_directories(paroli-daemon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

include(CTest)
if (BUILD_TESTING)
    if(DEFINED ENV{PAROLI_TEST_ENCODER} AND DEFINED ENV{PAROLI_TEST_DECODER} AND DEFINED ENV{PAROLI_TEST_CONFIG} AND DEFINED ENV{PAROLI_TEST_ESPEAK})
        add_test(NAME paroli-daemon-smoke
            COMMAND bash -lc "echo '{\"text\":\"test\",\"format\":\"wav\"}' | ./paroli-daemon --encoder $ENV{PAROLI_TEST_ENCODER} --decoder $ENV{PAROLI_TEST_DECODER} -c $ENV{PAROLI_TEST_CONFIG} --espeak_data $ENV{PAROLI_TEST_ESPEAK} | { read -n 1 b; test -n \"$b\"; }")
        set_tests_properties(paroli-daemon-smoke PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
endif()

