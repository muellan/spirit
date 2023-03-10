MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- ChooseCompiler.cmake ------------------------- <<" )
######## UI Web - this means we need emcc ###########################
if (SPIRIT_BUILD_FOR_JS)
    ###
    MESSAGE( STATUS ">> Choosing compiler:             emcc" )
    ### Set the path to emscripten
    # SET(EMSCRIPTEN_ROOT_PATH "/usr/local/emsdk/emscripten/1.38.29/")
    ### Use the Emscripten toolchain file
    # SET(CMAKE_TOOLCHAIN_FILE /usr/local/emsdk/emscripten/1.38.13/cmake/Modules/Platform/Emscripten)
    ###
    MESSAGE( STATUS ">> EMSCRIPTEN_ROOT_PATH: \"${EMSCRIPTEN_ROOT_PATH}\"")
    MESSAGE( STATUS ">> CMAKE_TOOLCHAIN_FILE: \"${CMAKE_TOOLCHAIN_FILE}\"")
######################################################################


######## Otherwise we can choose freely ##############################
else()
    if ( USER_COMPILER_C AND USER_COMPILER_CXX AND USER_PATH_COMPILER )
        MESSAGE( STATUS ">> User C compiler:           " ${USER_COMPILER_C} )
        MESSAGE( STATUS ">> User CXX compiler:         " ${USER_COMPILER_CXX} )
        MESSAGE( STATUS ">> User compiler path:        " ${USER_PATH_COMPILER} )
        set(CMAKE_C_COMPILER   ${USER_PATH_COMPILER}/${USER_COMPILER_C})
        set(CMAKE_CXX_COMPILER ${USER_PATH_COMPILER}/${USER_COMPILER_CXX})
    endif()
######################################################################

######## The End #####################################################
endif()
######################################################################
MESSAGE( STATUS ">> CMAKE_C_COMPILER:        " ${CMAKE_C_COMPILER} )
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER:      " ${CMAKE_CXX_COMPILER} )
MESSAGE( STATUS ">> --------------------- ChooseCompiler.cmake done -------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )