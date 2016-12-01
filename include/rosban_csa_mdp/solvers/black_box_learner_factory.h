#include "rosban_csa_mdp/solvers/black_box_learner.h"

#include "rosban_utils/factory.h"

namespace csa_mdp
{

class BlackBoxLearnerFactory : public rosban_utils::Factory<BlackBoxLearner> {
public:
  BlackBoxLearnerFactory();
};

}
