#include "starkit_csa_mdp/action_optimizers/action_optimizer_factory.h"

#include "starkit_csa_mdp/action_optimizers/basic_optimizer.h"

namespace csa_mdp
{
ActionOptimizerFactory::ActionOptimizerFactory()
{
  registerBuilder("BasicOptimizer", []() { return std::unique_ptr<ActionOptimizer>(new BasicOptimizer); });
}

}  // namespace csa_mdp
