project(cvwrap-vexcl)

set(SOURCE_FILES
    "pluginMain.cpp"
    "cvWrapCmd.cpp"
    "cvWrapCmd.h"
    "cvWrapDeformer.cpp"
    "cvWrapDeformer.h"
    "bindingio.cpp"
    "bindingio.h"
    "common.cpp"
    "common.h"
    "cvwrap.cl"
)

if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif ()

if (WIN32)
    set(COMPILE_FLAGS "/arch:AVX")
else()
    set(COMPILE_FLAGS "-mavx")
endif()

# find_package(
#     OpenCL
#     # 1.1 
#     # EXACT
# )
# todo resolve opencl conflicts

# find_package(
#     Boost
# )

# find_package(
#     VexCL
#     REQUIRED COMPONENTS
#     Compute
# )

ADD_LIBRARY(
    ${PROJECT_NAME}
    SHARED
    ${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES(
    ${PROJECT_NAME} 
    PRIVATE 
    VexCL::Compute
    # VexCL::OpenCL
    Maya::Maya
    MayaAPIUtils
)

TARGET_INCLUDE_DIRECTORIES(
    ${PROJECT_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    Maya::Maya
)

TARGET_COMPILE_OPTIONS(
    ${PROJECT_NAME}
    PRIVATE
    ${COMPILE_FLAGS}
)

# TARGET_COMPILE_DEFINITIONS(
#     ${PROJECT_NAME}
#     PUBLIC
#     __CL_ENABLE_EXCEPTIONS
#     CL_USE_DEPRECATED_OPENCL_1_1_APIS
# )

MAYA_PLUGIN(${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} ${MAYA_TARGET_TYPE} DESTINATION plug-ins)
install(FILES "cvwrap.cl" DESTINATION plug-ins)

