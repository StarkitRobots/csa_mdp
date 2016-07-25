#include "rosban_csa_mdp/solvers/learner.h"

namespace csa_mdp
{

void Learner::setStateLimits(const Eigen::VectorXd & new_state_limits)
{
  state_limits = new_state_limits;
}

void Learner::setActionLimits(const Eigen::VectorXd & new_action_limits)
{
  action_limits = new_action_limits;
}

Eigen::MatrixXd Learner::getStateLimits() const
{
  return state_limits;
}

Eigen::MatrixXd Learner::getActionLimits() const
{
  return action_limits;
}

void Learner::feed(const csa_mdp::Sample & sample)
{
  samples.push_back(sample);
}

}
