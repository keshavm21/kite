cmake_minimum_required(VERSION 3.16)
project(kite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)          # for clangd / VS Code
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

option(KITE_SANITIZE "Build with AddressSanitizer + UBSan" OFF)

# Warnings everywhere; no FMA contraction so the interpreter's float math matches
# PTX mul.rn/add.rn bit-for-bit (docs/TESTING.md section 3).
add_compile_options(-Wall -Wextra -Wpedantic -ffp-contract=off)
if(KITE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# Every .cpp under src/ except main.cpp goes into a library shared by kitec and the tests.
file(GLOB_RECURSE KITE_LIB_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(FILTER KITE_LIB_SOURCES EXCLUDE REGEX ".*/src/main\\.cpp$")
if(KITE_LIB_SOURCES)
  add_library(kite_lib STATIC ${KITE_LIB_SOURCES})
  target_include_directories(kite_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)
else()                                         # Milestone 0: only main.cpp exists yet
  add_library(kite_lib INTERFACE)
  target_include_directories(kite_lib INTERFACE ${PROJECT_SOURCE_DIR}/src)
endif()

add_executable(kitec src/main.cpp)
target_link_libraries(kitec PRIVATE kite_lib)

# Unit tests: every tests/unit/*.cpp, using the vendored single-header doctest.
enable_testing()
file(GLOB KITE_UNIT_TESTS CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/unit/*.cpp)
if(KITE_UNIT_TESTS)
  add_executable(unit_tests ${KITE_UNIT_TESTS})
  target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
  target_link_libraries(unit_tests PRIVATE kite_lib)
  add_test(NAME unit_tests COMMAND unit_tests)
endif()
