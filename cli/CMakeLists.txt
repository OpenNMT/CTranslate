find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(
  ${Boost_INCLUDE_DIRS}
  )

if (WIN32 AND NOT CYGWIN AND NOT Boost_USE_STATIC_LIBS)
  add_definitions(-DBOOST_ALL_NO_LIB) #Boost: Tells the config system not to automatically select which libraries to link against
  add_definitions(-DBOOST_ALL_DYN_LINK) #Boost: Forces all libraries that have separate source, to be linked as dll's rather than static libraries on Microsoft Windows
endif()

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
