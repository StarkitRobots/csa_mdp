#include "rhoban_csa_mdp/value_approximators/value_approximator.h"

#include "rhoban_utils/serialization/factory.h"

namespace csa_mdp
{
class ValueApproximatorFactory : public rhoban_utils::Factory<ValueApproximator>
{
public:
  ValueApproximatorFactory();
};

}  // namespace csa_mdp
