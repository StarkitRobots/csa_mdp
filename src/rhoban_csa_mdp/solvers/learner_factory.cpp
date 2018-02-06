#include "rhoban_csa_mdp/solvers/learner_factory.h"

#include "rhoban_csa_mdp/solvers/fake_learner.h"
#include "rhoban_csa_mdp/solvers/model_based_learner.h"
#include "rhoban_csa_mdp/solvers/mre.h"

namespace csa_mdp
{

LearnerFactory::LearnerFactory()
{
  registerBuilder("FakeLearner", [](){return std::unique_ptr<Learner>(new FakeLearner);});
  registerBuilder("ModelBasedLearner",
                  [](){return std::unique_ptr<Learner>(new ModelBasedLearner);});
  registerBuilder("MRE", [](){return std::unique_ptr<Learner>(new MRE);});
}

}
