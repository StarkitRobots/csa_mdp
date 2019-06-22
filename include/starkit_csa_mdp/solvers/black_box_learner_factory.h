#include "starkit_csa_mdp/solvers/black_box_learner.h"

#include "starkit_utils/serialization/factory.h"

namespace csa_mdp
{
class BlackBoxLearnerFactory : public starkit_utils::Factory<BlackBoxLearner>
{
public:
  BlackBoxLearnerFactory();
};

}  // namespace csa_mdp
