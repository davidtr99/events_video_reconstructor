cmake_minimum_required(VERSION 3.10)
project(events_video_reconstructor)
set(CMAKE_CXX_FLAGS "-std=c++17")
find_package(CUDA 11.7 REQUIRED)

# ==== Build type (default: release) ====
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)

IF(DEFINED CMAKE_BUILD_TYPE)
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
ENDIF()

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Release")
ENDIF()

IF(NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug" AND
    NOT ${CMAKE_BUILD_TYPE} STREQUAL "Release")
    MESSAGE(FATAL_ERROR "Only Release and Debug build types are allowed.")
ENDIF()

# ==== Find catkin macros and libraries ====
find_package(catkin REQUIRED COMPONENTS
  roscpp
  dvs_msgs
  roslib
)

# ==== Catkin setup ====
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp dvs_msgs roslib
)

# ==== Include directories ====
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# ==== Link directories ====
link_directories(
  include
  ${catkin_LIBRARY_DIRS}
)

# ==== Build ====
file(GLOB_RECURSE SRC_INCS "include/${PROJECT_NAME}/*.h*")
file(GLOB_RECURSE SRC_SRCS "src/*.c*")
file(GLOB_RECURSE SRC_HDRS "src/*.h*")

add_library(${PROJECT_NAME} ${SRC_SRCS} ${SRC_HDRS} ${SRC_INCS})
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${CUDA_LIBRARIES})

# ==== Loading 3rd party libraries ====
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
if(${Python3_FOUND})
    target_link_libraries(${PROJECT_NAME} PUBLIC Python3::Python)
    target_link_libraries(${PROJECT_NAME} PUBLIC Python3::NumPy)
endif(${Python3_FOUND})

find_package(pybind11 CONFIG REQUIRED)
if(${pybind11_FOUND})
    target_link_libraries(${PROJECT_NAME} PUBLIC pybind11::headers)
    target_link_libraries(${PROJECT_NAME} PUBLIC pybind11::embed)
endif(${pybind11_FOUND})

add_executable(${PROJECT_NAME}_node utils/events_video_reconstructor_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

# ==== Install ====
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)