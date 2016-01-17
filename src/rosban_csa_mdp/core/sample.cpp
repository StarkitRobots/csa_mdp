#include "rosban_csa_mdp/core/sample.h"

namespace csa_mdp
{

Sample::Sample()
  : reward(0)
{
}

Sample::Sample(const Eigen::VectorXd &state_,
               const Eigen::VectorXd &action_,
               const Eigen::VectorXd &next_state_,
               double reward_)
  : state(state_),
    action(action_),
    next_state(next_state_),
    reward(reward_)
{
}

}
