#include "rosban_csa_mdp/core/opportunist_policy.h"

#include "rosban_csa_mdp/core/policy_factory.h"
#include "rosban_csa_mdp/core/problem_factory.h"

namespace csa_mdp
{

OpportunistPolicy::OpportunistPolicy(){}

void OpportunistPolicy::setActionLimits(const std::vector<Eigen::MatrixXd> & limits)
{
  Policy::setActionLimits(limits);
  for (size_t policy_id = 0; policy_id < policies.size(); policy_id++) {
    policies[policy_id]->setActionLimits(limits);
  }
}

Eigen::VectorXd
OpportunistPolicy::getRawAction(const Eigen::VectorXd & state,
                                  std::default_random_engine * engine) const
{
  if (policies.size() == 0) {
    throw std::logic_error("OpportunistPolicy::getRawAction: no available policies");
  }
  double best_score = std::numeric_limits<double>::lowest();
  Eigen::VectorXd best_action;
  std::cout << "-------------" << std::endl;
  for (size_t policy_id = 0; policy_id < policies.size(); policy_id++) {
    double cumulated_reward = 0;
    const csa_mdp::Policy & policy = *(policies[policy_id]);
    for (int rollout = 0; rollout < nb_rollouts; rollout++) {
      cumulated_reward +=
        problem->sampleRolloutReward(state, policy, horizon, 1.0, engine);
    }
    double avg_reward = cumulated_reward / nb_rollouts;
    std::cout << "Best score for policy " << policy_id << ": "
              << avg_reward << std::endl;
    if (avg_reward > best_score) {
      std::cout << "<- best action" << std::endl;
      best_action = policy.getAction(state, engine);
      best_score = avg_reward;
    }
  }
  return best_action;
}

void OpportunistPolicy::to_xml(std::ostream & out) const
{
  (void) out;
  throw std::logic_error("OpportunistPolicy::to_xml: not implemented");
}

void OpportunistPolicy::from_xml(TiXmlNode * node)
{
  problem = ProblemFactory().read(node,"problem");
  policies = PolicyFactory().readVector(node,"policies");
  rosban_utils::xml_tools::try_read<int>(node, "nb_rollouts" , nb_rollouts);
  rosban_utils::xml_tools::try_read<int>(node, "horizon" , horizon);
  setActionLimits(problem->getActionsLimits());
}
std::string OpportunistPolicy::class_name() const
{
  return "OpportunistPolicy";
}


}
