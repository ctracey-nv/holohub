# TODO: pva-solutions should ship its own config.cmake
find_library(PVA_RADAR_LIB "radar_advanced_operators"
             PATHS "/opt/nvidia/pva-solutions-0.4/lib/x86_64-linux-gnu")
add_library(radar_advanced_operators SHARED IMPORTED)

set_target_properties(radar_advanced_operators PROPERTIES IMPORTED_LOCATION ${PVA_RADAR_LIB})
target_include_directories(radar_advanced_operators INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")

find_library(PVA_OPS_LIB "pva_operator" PATHS "/opt/nvidia/pva-solutions-0.4/lib/x86_64-linux-gnu")

add_library(pva_operator SHARED IMPORTED)
set_target_properties(pva_operator PROPERTIES IMPORTED_LOCATION ${PVA_OPS_LIB})
target_include_directories(pva_operator INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")

find_library(NVCV_TYPES_LIB "nvcv_types_d" PATHS "/opt/nvidia/pva-solutions-0.4/lib/x86_64-linux-gnu")

add_library(nvcv_types SHARED IMPORTED)
set_target_properties(nvcv_types PROPERTIES IMPORTED_LOCATION ${NVCV_TYPES_LIB})
target_include_directories(nvcv_types INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")