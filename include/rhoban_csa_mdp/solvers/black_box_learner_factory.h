#include "rhoban_csa_mdp/solvers/black_box_learner.h"

#include "rhoban_utils/serialization/factory.h"

namespace csa_mdp
{
class BlackBoxLearnerFactory : public rhoban_utils::Factory<BlackBoxLearner>
{
public:
  BlackBoxLearnerFactory();
};

}  // namespace csa_mdp
