message ("Helpers build.")

file(GLOB_RECURSE HELPER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library (Helpers ${HELPER_SOURCES})

if ("${CXX11_ENABLED}")
  target_compile_features (Helpers PUBLIC cxx_std_11)
endif ()
