set(EXECUTABLE cpu_verushash)

#cpu_verushash/
file(GLOB SRC_LIST
    cpu_verushash.cpp )
file(GLOB HEADERS
    cpu_verushash.hpp
    )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(..)
ADD_LIBRARY(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${EXECUTABLE} )

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
