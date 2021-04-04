cmake_minimum_required(VERSION 3.5) # may work with earlier versions

project(pymcas CXX)

option(BUILD_PYTHON_SUPPORT "Build Python support" OFF)

if (PYTHON AND BUILD_PYTHON_SUPPORT)

  # this is the directory for where the package will be constructed
  #
  set(PKG_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/omp")
  

  # copy python files to build
  
  file(MAKE_DIRECTORY ${PKG_SOURCE_DIR})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_BINARY_DIR}/LICENSE)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/algos.py ${PKG_SOURCE_DIR}/algos.py COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/models.py ${PKG_SOURCE_DIR}/models.py COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${PKG_SOURCE_DIR}/__init__.py COPYONLY)
  
  
  
else()
  message("-- NOT building Python support.")
endif()

