cmake_minimum_required(VERSION 3.5) # may work with earlier versions

project(omp CXX)

option(BUILD_PYTHON_SUPPORT "Build Python support" OFF)

if (PYTHON AND BUILD_PYTHON_SUPPORT)

  # this is the directory for where the package will be constructed
  #
  set(PKG_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/omp")
  
  get_filename_component(PYTHON_FILENAME ${PYTHON} NAME)
  # get python locations
  execute_process ( COMMAND ${PYTHON} -m site --user-site OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process ( COMMAND ${PYTHON} -c "from sysconfig import get_paths as gp; print(gp()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process ( COMMAND ${PYTHON} -c "from sysconfig import get_config_vars as gc; print(gc()['LIBPL'])" OUTPUT_VARIABLE PYTHON_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process ( COMMAND ${PYTHON} -c "from sysconfig import get_config_vars as gc; print(gc()['BLDLIBRARY'])" OUTPUT_VARIABLE PYTHON_BLDLIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  
  message("-- Python    site : ${PYTHON_SITE_PACKAGES}")
  message("--         stdlib : ${PYTHON_LIB_DIR}")
  message("--        include : ${PYTHON_INCLUDE_DIR}")
  

  message("-- Building Python support.")
  set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
  set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
 
  # copy python files to build
  
  file(MAKE_DIRECTORY ${PKG_SOURCE_DIR})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_BINARY_DIR}/LICENSE)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/algos.py ${PKG_SOURCE_DIR}/algos.py COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/models.py ${PKG_SOURCE_DIR}/models.py COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${PKG_SOURCE_DIR}/__init__.py COPYONLY)
  
  add_custom_target(${PROJECT_NAME}-wheel COMMAND ${PYTHON} ${SETUP_PY} bdist) # build wheel binary distribution
  # install as a site-package
  install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --user)")
  
else()
  message("-- NOT building Python support.")
endif()

