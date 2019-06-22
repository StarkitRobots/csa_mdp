#include "starkit_csa_mdp/value_approximators/value_approximator.h"

#include "starkit_utils/serialization/factory.h"

namespace csa_mdp
{
class ValueApproximatorFactory : public starkit_utils::Factory<ValueApproximator>
{
public:
  ValueApproximatorFactory();
};

}  // namespace csa_mdp
