message ("Solver build.")

file(GLOB_RECURSE SOLVER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library (Solver ${SOLVER_SOURCES})
