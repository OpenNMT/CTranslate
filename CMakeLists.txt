cmake_minimum_required(VERSION 3.1.0)

# Set policy for setting the MSVC runtime library for static MSVC builds
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(onmt)

option(LIB_ONLY "Do not compile clients" OFF)
option(WITH_OPENMP "Use OpenMP if available" ON)
option(WITH_CUDA "Use CUDA if available" ON)
option(WITH_MKL "Use Intel® MKL if available" ON)
option(WITH_QLINEAR "Add Quantized linear module, value can be OFF, SSE, AVX2 or AVX512" OFF)
option(WITH_BOOST_LOG "Use Boost log if available" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(CMAKE_CXX_STANDARD 11)

# Set Release build type by default to get sane performance.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

if(WITH_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    message(WARNING "OpenMP not found: Compilation will not use OpenMP")
  endif()
endif()

if(ANDROID)
  set(LIB_ONLY ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  if(ANDROID_STL MATCHES "gnustl")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_GNUSTL_COMPAT")
  endif()
endif()

if(MSVC)
  if(NOT BUILD_SHARED_LIBS)
    if(CMAKE_VERSION VERSION_LESS "3.15.0")
      message(FATAL_ERROR "Use CMake 3.15 or later when setting BUILD_SHARED_LIBS to OFF")
    endif()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()

add_subdirectory(lib/TH)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if (EIGEN_ROOT)
  set(ENV{EIGEN3_ROOT} ${EIGEN_ROOT})
endif()
if (EIGEN3_ROOT)
  set(ENV{EIGEN3_ROOT} ${EIGEN3_ROOT})
endif()

find_package(Eigen3 3.3 REQUIRED)
if(WITH_CUDA)
  find_package(CUDA 6.5)
endif()
if(WITH_MKL)
  find_package(MKL)
endif()
if(WITH_BOOST_LOG)
  find_package(Boost COMPONENTS log)
endif()

set(INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_BINARY_DIR}
  )
set(SOURCES
  src/th/Env.cc
  src/th/Obj.cc
  src/th/Utils.cc
  src/Dictionary.cc
  src/SubDict.cc
  src/PhraseTable.cc
  src/Profiler.cc
  src/ITranslator.cc
  src/TranslatorFactory.cc
  src/TranslationOptions.cc
  src/TranslationResult.cc
  src/Threads.cc
  )
set(LIBRARIES
  OpenNMTTokenizer
  TH)

find_library(ONMT_TOKENIZER_LIBRARY NAMES OpenNMTTokenizer)
find_path(ONMT_TOKENIZER_INCLUDE_DIR NAMES onmt/Tokenizer.h)

if(NOT ONMT_TOKENIZER_LIBRARY OR NOT ONMT_TOKENIZER_INCLUDE_DIR)
  message(STATUS "Using OpenNMT tokenizer submodule")
  add_subdirectory(lib/tokenizer)
  list(APPEND INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/lib/tokenizer/include)
else()
  message(STATUS "Found OpenNMT tokenizer: ${ONMT_TOKENIZER_LIBRARY}")
  list(APPEND INCLUDE_DIRECTORIES ${ONMT_TOKENIZER_INCLUDE_DIR})
  get_filename_component(ONMT_TOKENIZER_LIBRARY_DIR ${ONMT_TOKENIZER_LIBRARY} DIRECTORY)
  link_directories(${ONMT_TOKENIZER_LIBRARY_DIR})
endif()

if (WITH_QLINEAR)
  IF(WITH_QLINEAR STREQUAL "AVX2")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_QLINEAR -DSIMD_AVX2")
      LIST(APPEND SOURCES
        src/simd/AVX2_MatrixMult.cc
      )
    MESSAGE(STATUS "Using AVX2 instruction sets")
  ELSEIF(WITH_QLINEAR STREQUAL "SSE")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_QLINEAR -DSIMD_SSE")
      LIST(APPEND SOURCES
        src/simd/SSE_MatrixMult.cc
      )
    MESSAGE(STATUS "Using SSE instruction sets")
  ELSEIF(WITH_QLINEAR STREQUAL "AVX512")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_QLINEAR -DSIMD_AVX512")
      LIST(APPEND SOURCES
        src/simd/AVX512_MatrixMult.cc
      )
    MESSAGE(STATUS "Using AVX512 instruction sets")
  ELSE(WITH_QLINEAR STREQUAL "AVX2")
      MESSAGE(FATAL_ERROR "Incorrect value of WITH_QLINEAR option")
  ENDIF(WITH_QLINEAR STREQUAL "AVX2")
endif()

if (MKL_FOUND)
  add_definitions(-DWITH_MKL)
  add_definitions(-DEIGEN_USE_MKL_ALL)
  list(APPEND INCLUDE_DIRECTORIES ${MKL_INCLUDE_DIR})
  list(APPEND LIBRARIES ${MKL_LIBRARIES})
endif()

if (Boost_LOG_FOUND)
  add_definitions(-DWITH_BOOST_LOG)
  list(APPEND INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
  list(APPEND LIBRARIES ${Boost_LIBRARIES})
  list(APPEND SOURCES src/Logger.cc)
  if (NOT Boost_USE_STATIC_LIBS)
    add_definitions(-DBOOST_ALL_DYN_LINK)
    if (WIN32 AND NOT CYGWIN)
      add_definitions(-DBOOST_ALL_NO_LIB)
    endif()
  endif()
endif()

if (CUDA_FOUND)
  add_definitions(-DWITH_CUDA)
  if(MSVC)
    if(BUILD_SHARED_LIBS)
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/MD$<$<CONFIG:Debug>:d>")
    else()
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/MT$<$<CONFIG:Debug>:d>")
    endif()
  endif()
  cuda_include_directories(${INCLUDE_DIRECTORIES})
  cuda_add_library(${PROJECT_NAME}
    ${SOURCES}
    src/cuda/Utils.cc
    src/cuda/Kernels.cu
    )
  cuda_add_cublas_to_target(${PROJECT_NAME})
else()
  add_library(${PROJECT_NAME} ${SOURCES})
endif()

include(GNUInstallDirs)
include(GenerateExportHeader)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
generate_export_header(${PROJECT_NAME} EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/onmt/${PROJECT_NAME_LOWER}_export.h)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRECTORIES})

if (MSVC AND NOT (MSVC_VERSION LESS 1910))
  if (Boost_LOG_FOUND)
    target_compile_options(${PROJECT_NAME} PUBLIC /permissive-) #to avoid C4596 errors in VS2017+
  endif()
  if (OPENMP_FOUND)
    target_compile_options(${PROJECT_NAME} PUBLIC /Zc:twoPhase-) #to avoid "C2338: two-phase name lookup is not supported for C++/CLI, C++/CX, or OpenMP; use /Zc:twoPhase-" errors in VS2017+
  endif()
endif()

if (NOT LIB_ONLY)
  add_subdirectory(cli)
endif()

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/onmt"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.h*" PATTERN "*.cuh"
  )
install(
  FILES "${PROJECT_BINARY_DIR}/onmt/${PROJECT_NAME_LOWER}_export.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onmt"
  )
