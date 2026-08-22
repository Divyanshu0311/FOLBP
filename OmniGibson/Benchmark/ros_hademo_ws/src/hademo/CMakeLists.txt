cmake_minimum_required(VERSION 3.8)
project(hademo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Order matters only for readability; rosidl resolves intra-package deps itself.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Args.msg"
  "msg/FuncAndArgs.msg"
  "msg/Action.msg"
  "msg/ResultInfo.msg"
  "msg/Result.msg"
  DEPENDENCIES std_msgs
)

# The two nodes are plain scripts (sim.py imports action_subscriber.py by path
# from the source tree), so install them as programs rather than as a python
# package.
install(PROGRAMS
  src/action_publisher.py
  src/action_subscriber.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
