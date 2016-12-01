#include "rosban_csa_mdp/solvers/black_box_learner_factory.h"

namespace csa_mdp
{

BlackBoxLearnerFactory::BlackBoxLearnerFactory() {
  registerBuilder("BlackBoxLearner",
                  [](){return std::unique_ptr<BlackBoxLearner>(new BlackBoxLearner);});
}

}
