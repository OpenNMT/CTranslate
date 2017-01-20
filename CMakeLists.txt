cmake_minimum_required(VERSION 3.1.0)
project(onmt)

option(LIB_ONLY "Do not compile clients" OFF)

set(CMAKE_CXX_STANDARD 11)

if(NOT Boost_FOUND)
  find_package(Boost)
endif()

add_subdirectory(lib/TH)

set(INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/TH
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN_ROOT}/include/eigen3
add_subdirectory(lib/unicode)
  ${Boost_INCLUDE_DIRS}
  )
include_directories(${INCLUDE_DIRECTORIES})

  ${CMAKE_CURRENT_SOURCE_DIR}/lib/unicode

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fopenmp -DNDEBUG")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  string(REPLACE "-fopenmp" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

if(ANDROID)
  set(LIB_ONLY ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  if (ANDROID_STL MATCHES "gnustl")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_GNUSTL_COMPAT")
  endif()
endif()

target_link_libraries(${PROJECT_NAME}
  unicode
  TH)

target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRECTORIES})

if (NOT LIB_ONLY)
  add_subdirectory(cli)
endif()
