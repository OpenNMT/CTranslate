# Modified from Caffe.


# All contributions by the University of California:
# Copyright (c) 2014-2017 The Regents of the University of California (Regents)
# All rights reserved.

# All other contributions:
# Copyright (c) 2014-2017, the respective contributors
# All rights reserved.

# Caffe uses a shared copyright model: each contributor holds copyright over
# their contributions to Caffe. The project versioning records all such
# contribution and copyright details. If a contributor wants to further mark
# their specific copyright on a particular contribution, they should indicate
# their copyright solely in the commit message of the change when it is
# committed.


# Find the MKL libraries
#
# Options:
#
#   MKL_USE_STATIC_LIBS             : use static libraries
#
# This module defines the following variables:
#
#   MKL_FOUND            : True mkl is found
#   MKL_INCLUDE_DIR      : unclude directory
#   MKL_LIBRARIES        : the libraries to link against.


# ---[ Options
option(MKL_USE_STATIC_LIBS "Use static libraries" OFF)

# ---[ Root folders
if(WIN32)
  set(ProgramFilesx86 "ProgramFiles(x86)")
  set(INTEL_ROOT_DEFAULT $ENV{${ProgramFilesx86}}/IntelSWTools/compilers_and_libraries/windows)
else()
  set(INTEL_ROOT_DEFAULT "/opt/intel")
endif()
set(INTEL_ROOT ${INTEL_ROOT_DEFAULT} CACHE PATH "Folder contains intel libs")
find_path(MKL_ROOT include/mkl.h PATHS $ENV{MKLROOT} ${INTEL_ROOT}/mkl
                                   DOC "Folder contains MKL")

# ---[ Find include dir
find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT} PATH_SUFFIXES include)
set(__looked_for MKL_INCLUDE_DIR)

# ---[ Find libraries
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(__path_suffixes lib lib/ia32)
else()
  set(__path_suffixes lib lib/intel64)
endif()

set(__mkl_libs "")

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  if(WIN32)
    list(APPEND __mkl_libs intel_c)
  else()
    list(APPEND __mkl_libs intel)
  endif()
else()
  list(APPEND __mkl_libs intel_lp64)
endif()

if(WIN32)
  list(APPEND __mkl_libs intel_thread)
else()
  list(APPEND __mkl_libs gnu_thread)
endif()

list(APPEND __mkl_libs core)

foreach (__lib ${__mkl_libs})
  set(__mkl_lib "mkl_${__lib}")
  string(TOUPPER ${__mkl_lib} __mkl_lib_upper)

  if(WIN32)
    if(NOT MKL_USE_STATIC_LIBS)
      set(__mkl_lib "${__mkl_lib}_dll")
    endif()
  else()
    if(MKL_USE_STATIC_LIBS)
      set(__mkl_lib "lib${__mkl_lib}.a")
    endif()
  endif()

  find_library(${__mkl_lib_upper}_LIBRARY
        NAMES ${__mkl_lib}
        PATHS ${MKL_ROOT} "${MKL_INCLUDE_DIR}/.."
        PATH_SUFFIXES ${__path_suffixes}
        DOC "The path to Intel(R) MKL ${__mkl_lib} library")
  mark_as_advanced(${__mkl_lib_upper}_LIBRARY)

  list(APPEND __looked_for ${__mkl_lib_upper}_LIBRARY)
  list(APPEND MKL_LIBRARIES ${${__mkl_lib_upper}_LIBRARY})
endforeach()

if(WIN32)
  set(__iomp5_libs iomp5 libiomp5md.lib)
  find_library(MKL_RTL_LIBRARY ${__iomp5_libs}
    PATHS ${INTEL_ROOT} ${INTEL_ROOT}/compiler ${MKL_ROOT}/.. ${MKL_ROOT}/../compiler
    PATH_SUFFIXES ${__path_suffixes}
    DOC "Path to OpenMP runtime library")

  list(APPEND __looked_for MKL_RTL_LIBRARY)
  list(APPEND MKL_LIBRARIES ${MKL_RTL_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG ${__looked_for})

if(MKL_FOUND)
  message(STATUS "Found MKL (include: ${MKL_INCLUDE_DIR}, lib: ${MKL_LIBRARIES}")
endif()
