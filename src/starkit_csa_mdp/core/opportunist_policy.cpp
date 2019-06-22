#include "starkit_csa_mdp/core/opportunist_policy.h"

#include "starkit_csa_mdp/core/policy_factory.h"
#include "starkit_csa_mdp/core/problem_factory.h"

namespace csa_mdp
{
OpportunistPolicy::OpportunistPolicy()
{
}

void OpportunistPolicy::setActionLimits(const std::vector<Eigen::MatrixXd>& limits)
{
  Policy::setActionLimits(limits);
  for (size_t policy_id = 0; policy_id < policies.size(); policy_id++)
  {
    policies[policy_id]->setActionLimits(limits);
  }
}

Eigen::VectorXd OpportunistPolicy::getRawAction(const Eigen::VectorXd& state, std::default_random_engine* engine) const
{
  if (policies.size() == 0)
  {
    throw std::logic_error("OpportunistPolicy::getRawAction: no available policies");
  }
  double best_score = std::numeric_limits<double>::lowest();
  Eigen::VectorXd best_action;
  std::cout << "-------------" << std::endl;
  for (size_t policy_id = 0; policy_id < policies.size(); policy_id++)
  {
    double cumulated_reward = 0;
    const csa_mdp::Policy& policy = *(policies[policy_id]);
    for (int rollout = 0; rollout < nb_rollouts; rollout++)
    {
      cumulated_reward += problem->sampleRolloutReward(state, policy, horizon, 1.0, engine);
    }
    double avg_reward = cumulated_reward / nb_rollouts;
    std::cout << "Best score for policy " << policy_id << ": " << avg_reward << std::endl;
    if (avg_reward > best_score)
    {
      std::cout << "<- best action" << std::endl;
      best_action = policy.getAction(state, engine);
      best_score = avg_reward;
    }
  }
  return best_action;
}

Json::Value OpportunistPolicy::toJson() const
{
  throw std::logic_error("OpportunistPolicy::toJson: not implemented");
}

void OpportunistPolicy::fromJson(const Json::Value& v, const std::string& dir_name)
{
  problem = ProblemFactory().read(v, "problem", dir_name);
  policies = PolicyFactory().readVector(v, "policies", dir_name);
  starkit_utils::tryRead(v, "nb_rollouts", &nb_rollouts);
  starkit_utils::tryRead(v, "horizon", &horizon);
  setActionLimits(problem->getActionsLimits());
}
std::string OpportunistPolicy::getClassName() const
{
  return "OpportunistPolicy";
}

}  // namespace csa_mdp
