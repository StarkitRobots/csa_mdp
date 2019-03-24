#pragma once

#include "rhoban_utils/serialization/factory.h"

namespace csa_mdp
{
class ActionOptimizer;

class ActionOptimizerFactory : public rhoban_utils::Factory<ActionOptimizer>
{
public:
  ActionOptimizerFactory();
};

}  // namespace csa_mdp
