#include "rosban_csa_mdp/solvers/learner_factory.h"

#include "rosban_csa_mdp/solvers/model_based_learner.h"

namespace csa_mdp
{

LearnerFactory::LearnerFactory()
{
  registerBuilder("ModelBasedLearner",
                  [](TiXmlNode * node)
                  {
                    Learner * learner = new ModelBasedLearner();
                    learner->from_xml(node);
                    return learner;
                  });
}

}
