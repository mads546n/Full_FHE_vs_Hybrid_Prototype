cmake_minimum_required(VERSION 3.15)

project(HoEn_proto
        VERSION 0.1
        LANGUAGES CXX)

# -------------------------
# C++ standard
# -------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------
# Build type default
# -------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -------------------------
# Find Microsoft SEAL
# -------------------------
# Expect SEALConfig.cmake via -DSEAL_DIR=...
find_package(SEAL REQUIRED)

message(STATUS "Found Microsoft SEAL")
message(STATUS "SEAL version: ${SEAL_VERSION}")

# -------------------------
# Target
# -------------------------
add_executable(he_proto
        main.cpp
)

target_link_libraries(he_proto
        PRIVATE
        SEAL::seal
)

# -------------------------
# Warnings (nice to have)
# -------------------------
if (MSVC)
    target_compile_options(he_proto PRIVATE /W4)
else()
    target_compile_options(he_proto PRIVATE -Wall -Wextra -Wpedantic)
endif()
