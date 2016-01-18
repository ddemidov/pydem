#-----------------------------------------------------------------------------
set(DEM_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling the library")

# Try to autodetect Python (can be overridden manually if needed)
set(Python_ADDITIONAL_VERSIONS 3.4 3.5 3.6)
find_package(PythonLibs ${DEM_PYTHON_VERSION} REQUIRED)

# Enable C++11 mode on C++ / Clang
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -flto -Wno-deprecated-declarations")

# Include path for Python header files
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
include_directories(${PROJECT_SOURCE_DIR}/vexcl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#----------------------------------------------------------------------------
# Find OpenCL
#----------------------------------------------------------------------------
find_package(OpenCL REQUIRED)
include_directories( ${OpenCL_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# VexCL options
#----------------------------------------------------------------------------
add_definitions(-DBOOST_PROTO_MAX_ARITY=15)

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
find_package(Boost COMPONENTS system filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )

#-----------------------------------------------------------------------------
# Create the binding library
#-----------------------------------------------------------------------------
add_library(pydem_ext SHARED pydem.cpp)
target_link_libraries(pydem_ext ${Boost_LIBRARIES} ${OpenCL_LIBRARY})
set_target_properties(pydem_ext PROPERTIES PREFIX "")

