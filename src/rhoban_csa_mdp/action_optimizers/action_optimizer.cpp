#include "rhoban_csa_mdp/action_optimizers/action_optimizer.h"

namespace csa_mdp
{
ActionOptimizer::ActionOptimizer() : nb_threads(1)
{
}

ActionOptimizer::~ActionOptimizer()
{
}

void ActionOptimizer::setNbThreads(int new_nb_threads)
{
  nb_threads = new_nb_threads;
}

}  // namespace csa_mdp
