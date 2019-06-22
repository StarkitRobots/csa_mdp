cmake_minimum_required(VERSION 2.8.3)
project(starkit_csa_mdp)

find_package(catkin REQUIRED COMPONENTS
  starkit_fa
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++17")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES starkit_csa_mdp
  CATKIN_DEPENDS starkit_fa
)

## Gaussian Processes support
# Currently, uses of GP in CSA_MDP is directly linked with the use of GP in
# regression forests. It would be possible to separate those uses if necessary
# in the future.
if ("${catkin_LIBRARIES}" MATCHES "starkit_gp")
  add_definitions(-DSTARKIT_RF_USES_GP)
endif()


###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  )

# Directories containing sources files
set(DIRECTORIES
  src/starkit_csa_mdp/action_optimizers
  src/starkit_csa_mdp/core
  src/starkit_csa_mdp/knownness
  src/starkit_csa_mdp/online_planners
  src/starkit_csa_mdp/reward_predictors
  src/starkit_csa_mdp/value_approximators
  src/starkit_csa_mdp/solvers
  src/kd_trees# Should be elsewhere...
  )

# Build ALL_SOURCES
set (SOURCES)
set (ALL_SOURCES)
set (PREFIXED_SOURCES)
foreach (DIRECTORY ${DIRECTORIES})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}")
    include (${DIRECTORY}/sources.cmake)

    foreach (SOURCE ${SOURCES})
        set (PREFIXED_SOURCES ${PREFIXED_SOURCES} ${DIRECTORY}/${SOURCE})
    endforeach (SOURCE)

    set (ALL_SOURCES ${ALL_SOURCES} ${PREFIXED_SOURCES})
endforeach (DIRECTORY)

# Declare a C++ library
add_library(starkit_csa_mdp
  ${ALL_SOURCES}
  )

target_link_libraries(starkit_csa_mdp
  ${catkin_LIBRARIES}
  )
