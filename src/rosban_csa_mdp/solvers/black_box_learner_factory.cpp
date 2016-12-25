#include "rosban_csa_mdp/solvers/black_box_learner_factory.h"

#include "rosban_csa_mdp/solvers/policy_mutation_learner.h"
#include "rosban_csa_mdp/solvers/tree_policy_iteration.h"

namespace csa_mdp
{

BlackBoxLearnerFactory::BlackBoxLearnerFactory() {
  registerBuilder("TreePolicyIteration",
                  [](){return std::unique_ptr<TreePolicyIteration>(new TreePolicyIteration);});
  registerBuilder("PolicyMutationLearner",
                  [](){return std::unique_ptr<PolicyMutationLearner>(new PolicyMutationLearner);});
}

}
