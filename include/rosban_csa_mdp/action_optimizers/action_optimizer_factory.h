#pragma once

#include "rosban_utils/factory.h"

namespace csa_mdp
{

class ActionOptimizer;

class ActionOptimizerFactory : public rosban_utils::Factory<ActionOptimizer>
{
public:
  ActionOptimizerFactory();
};

}
