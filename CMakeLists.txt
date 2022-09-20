cmake_minimum_required(VERSION 3.0.2)
project(ioc_srv)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
  )

add_service_files(
   FILES
   ioc.srv
 )

generate_messages(
  DEPENDENCIES
  std_msgs
)

