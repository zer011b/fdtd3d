message ("Scheme build.")

file(GLOB_RECURSE INTERNAL_SCHEME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/InternalScheme*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/CallBack*.cpp")
file(GLOB_RECURSE CUDA_INTERNAL_SCHEME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/InternalScheme*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/CallBack*.cu")

file(GLOB_RECURSE SCHEME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Scheme*.cpp")

if ("${CUDA_ENABLED}")
  list(FILTER INTERNAL_SCHEME_SOURCES EXCLUDE REGEX ".*/InternalScheme*.cpp$")
  list(FILTER INTERNAL_SCHEME_SOURCES EXCLUDE REGEX ".*/CallBack.cpp$")
  add_library (InternalScheme ${INTERNAL_SCHEME_SOURCES} ${CUDA_INTERNAL_SCHEME_SOURCES})
  set_target_properties (InternalScheme PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_library (InternalScheme ${INTERNAL_SCHEME_SOURCES})
endif ()

add_library (Scheme ${SCHEME_SOURCES})

if ("${CXX11_ENABLED}")
  target_compile_features (InternalScheme PUBLIC cxx_std_11)
  target_compile_features (Scheme PUBLIC cxx_std_11)
endif ()

target_link_libraries (InternalScheme Layout Settings Grid)
target_link_libraries (Scheme InternalScheme Dumper Loader)
