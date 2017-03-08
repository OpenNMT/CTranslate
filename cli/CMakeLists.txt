find_package(Boost COMPONENTS program_options)

include_directories(
  ${Boost_INCLUDE_DIRS}
  )

add_executable(translate
  translate.cc
  BatchReader.cc
  BatchWriter.cc
  )
target_link_libraries(translate
  ${PROJECT_NAME}
  ${Boost_LIBRARIES}
  )

install(
  TARGETS translate
  DESTINATION bin/
  )
