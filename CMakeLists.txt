project(cvwrap-reference)

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

ADD_LIBRARY(
    ${PROJECT_NAME}
    SHARED
    ${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES(
    ${PROJECT_NAME} 
    PRIVATE 
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

MAYA_PLUGIN(${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} ${MAYA_TARGET_TYPE} DESTINATION plug-ins)
install(FILES "cvwrap.cl" DESTINATION plug-ins)

