message ("Settings build.")

file(GLOB_RECURSE SETTINGS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if ("${CUDA_ENABLED}")
  cuda_add_library (Settings ${SETTINGS_SOURCES})
else ()
  add_library (Settings ${SETTINGS_SOURCES})
endif ()

target_link_libraries (Settings Helpers)
