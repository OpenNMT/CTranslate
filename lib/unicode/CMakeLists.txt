include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

add_library(unicode SHARED
  unicode.cc
  unicode_data.cc
  )
