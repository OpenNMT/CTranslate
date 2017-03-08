include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

set(PUBLIC_HEADERS
  THGeneral.h
  THFile.h
  THDiskFile.h
  )

add_library(TH
  THGeneral.c
  THFile.c
  THDiskFile.c
  )

install(
  TARGETS TH
  ARCHIVE DESTINATION lib/
  LIBRARY DESTINATION lib/
  )
install(
  FILES ${PUBLIC_HEADERS}
  DESTINATION include/TH
  )
