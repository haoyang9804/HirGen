cmake_minimum_required(VERSION 3.2)

# set(CMAKE_C_COMPILER /usr/bin/clang)
# set(CMAKE_CXX_COMPILER /usr/bin/clang++)
# set(CMAKE_C_COMPILER /home/gakki/llvm-project/build/bin/clang)
# set(CMAKE_CXX_COMPILER /home/gakki/llvm-project/build/bin/clang++)
# if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#     set(CMAKE_C_COMPILER /opt/homebrew/Cellar/gcc/12.1.0/bin/gcc CACHE PATH "" FORCE)
#     set(CMAKE_CXX_COMPILER /opt/homebrew/Cellar/gcc/12.1.0/bin/g++ CACHE PATH "" FORCE)
# endif()
set (CMAKE_CXX_STANDARD 17)
project(hirgen C CXX)

file(GLOB_RECURSE SRCS src/*cpp)
message(${SRCS})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_executable(hirgen ${SRCS})
target_include_directories(hirgen
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)
