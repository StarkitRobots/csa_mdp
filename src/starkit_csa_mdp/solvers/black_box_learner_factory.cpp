#include "starkit_csa_mdp/solvers/black_box_learner_factory.h"

#include "starkit_csa_mdp/solvers/lppi.h"
#include "starkit_csa_mdp/solvers/policy_mutation_learner.h"
#include "starkit_csa_mdp/solvers/pml2.h"
#include "starkit_csa_mdp/solvers/tree_policy_iteration.h"

namespace csa_mdp
{
BlackBoxLearnerFactory::BlackBoxLearnerFactory()
{
  registerBuilder("TreePolicyIteration",
                  []() { return std::unique_ptr<TreePolicyIteration>(new TreePolicyIteration); });
  registerBuilder("PolicyMutationLearner",
                  []() { return std::unique_ptr<PolicyMutationLearner>(new PolicyMutationLearner); });
  registerBuilder("PML2", []() { return std::unique_ptr<PML2>(new PML2); });
  registerBuilder("LPPI", []() { return std::unique_ptr<LPPI>(new LPPI); });
}

}  // namespace csa_mdp
