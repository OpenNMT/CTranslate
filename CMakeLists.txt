cmake_minimum_required(VERSION 3.1.0)
project(onmt)

option(LIB_ONLY "Do not compile clients" OFF)
option(WITH_OPENMP "Use OpenMP if available" ON)

set(CMAKE_CXX_STANDARD 11)

if(NOT Boost_FOUND)
  find_package(Boost)
endif()

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

add_subdirectory(lib/TH)

set(INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/TH
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN_ROOT}/include/eigen3
  ${Boost_INCLUDE_DIRS}
  )
include_directories(${INCLUDE_DIRECTORIES})

add_library(${PROJECT_NAME} SHARED
  src/th/Env.cc
  src/th/Obj.cc
  src/th/Utils.cc
  src/Dictionary.cc
  src/PhraseTable.cc
  src/TranslatorFactory.cc
  src/TranslationResult.cc
  src/Threads.cc
  )

target_link_libraries(${PROJECT_NAME} TH)

target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRECTORIES})

if (NOT LIB_ONLY)
  add_subdirectory(cli)
endif()
