message ("Source build.")

include_directories ("${PROJECT_SOURCE_DIR}/Source/Helpers")

include_directories ("${PROJECT_SOURCE_DIR}/Source/Physics")

include_directories ("${PROJECT_SOURCE_DIR}/Source/Kernels")
set (LIBS ${LIBS} Kernels)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Settings/")
set (LIBS ${LIBS} Settings)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Coordinate")
set (LIBS ${LIBS} Coordinate)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Grid")
set (LIBS ${LIBS} Grid)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Layout")
set (LIBS ${LIBS} Layout)

include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/Loader")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/Dumper")
set (LIBS ${LIBS} Loader)
set (LIBS ${LIBS} Dumper)

add_executable (unit-test-parallel-grid unit-test-parallel-grid.cpp)
target_link_libraries (unit-test-parallel-grid ${LIBS} Helpers Settings)

add_executable (unit-test-grid unit-test-grid.cpp)
target_link_libraries (unit-test-grid ${LIBS} Helpers Settings)

add_executable (unit-test-dumpers-loaders unit-test-dumpers-loaders.cpp)
target_link_libraries (unit-test-dumpers-loaders ${LIBS} Helpers Settings)

add_executable (unit-test-coordinate unit-test-coordinate.cpp)
target_link_libraries (unit-test-coordinate Helpers Settings Coordinate)
