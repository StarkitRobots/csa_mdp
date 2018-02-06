#include "rhoban_csa_mdp/value_approximators/value_approximator_factory.h"

#include "rhoban_csa_mdp/value_approximators/extra_trees_approximator.h"

namespace csa_mdp
{

ValueApproximatorFactory::ValueApproximatorFactory() {
  registerBuilder("ExtraTreesApproximator",
                  [](){return std::unique_ptr<ValueApproximator>(new ExtraTreesApproximator);});
}

}
