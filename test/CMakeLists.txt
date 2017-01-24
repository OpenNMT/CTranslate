enable_testing()

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

add_subdirectory(tokenization)
