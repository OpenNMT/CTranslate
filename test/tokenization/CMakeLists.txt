file(GLOB files "*.raw")
foreach(file ${files})
  add_test(${file} sh ${CMAKE_CURRENT_SOURCE_DIR}/test.sh ${file} ${CMAKE_BINARY_DIR})
endforeach()
