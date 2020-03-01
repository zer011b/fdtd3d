file(GLOB_RECURSE SETTINGS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE CUDA_SETTINGS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

if ("${CUDA_ENABLED}")
  list(FILTER SETTINGS_SOURCES EXCLUDE REGEX ".*/Settings.cpp$")
  add_library (Settings ${SETTINGS_SOURCES} ${CUDA_SETTINGS_SOURCES})
  set_target_properties (Settings PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_library (Settings ${SETTINGS_SOURCES})
endif ()

target_link_libraries (Settings Helpers)
