#pragma once

#include "starkit_utils/serialization/factory.h"

namespace csa_mdp
{
class ActionOptimizer;

class ActionOptimizerFactory : public starkit_utils::Factory<ActionOptimizer>
{
public:
  ActionOptimizerFactory();
};

}  // namespace csa_mdp
