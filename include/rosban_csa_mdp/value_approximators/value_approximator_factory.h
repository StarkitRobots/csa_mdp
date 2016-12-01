#include "rosban_csa_mdp/value_approximators/value_approximator.h"

#include "rosban_utils/factory.h"

namespace csa_mdp
{

class ValueApproximatorFactory : public rosban_utils::Factory<ValueApproximator> {
public:
  ValueApproximatorFactory();
};

}
