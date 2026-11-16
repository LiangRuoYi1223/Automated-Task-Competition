cmake_minimum_required(VERSION 3.0.2)
project(treasure_auto)
find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS
  scripts/device_preflight.py
  scripts/run_match_offline.py
  scripts/vision_workbench.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config examples launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
