include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

add_library(TH SHARED
  THGeneral.c
  THFile.c
  THDiskFile.c
  )
