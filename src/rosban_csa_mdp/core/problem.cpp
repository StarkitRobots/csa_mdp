#include "rosban_csa_mdp/core/problem.h"

#include <chrono>

namespace csa_mdp
{

Problem::Problem()
{
  unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
  random_engine = std::default_random_engine(seed);
}

int Problem::stateDims() const
{
  return state_limits.rows();
}

int Problem::actionDims() const
{
  return action_limits.rows();
}

const Eigen::MatrixXd & Problem::getStateLimits() const
{
  return state_limits;
}

const Eigen::MatrixXd & Problem::getActionLimits() const
{
  return action_limits;
}

void Problem::setStateLimits(const Eigen::MatrixXd & new_limits)
{
  state_limits = new_limits;
  state_distribution.clear();
  for (int row = 0; row < new_limits.rows(); row++)
  {
    double min = new_limits(row, 0);
    double max = new_limits(row, 1);
    state_distribution.push_back(std::uniform_real_distribution<double>(min, max));
  }
}

void Problem::setActionLimits(const Eigen::MatrixXd & new_limits)
{
  action_limits = new_limits;
  action_distribution.clear();
  for (int row = 0; row < new_limits.rows(); row++)
  {
    double min = new_limits(row, 0);
    double max = new_limits(row, 1);
    action_distribution.push_back(std::uniform_real_distribution<double>(min, max));
  }
}

Eigen::VectorXd Problem::getRandomAction()
{
  Eigen::VectorXd action(actionDims());
  for (int i = 0; i < actionDims(); i++)
  {
    action(i) = action_distribution[i](random_engine);
  }
  return action;
}

Sample Problem::getRandomSample(const Eigen::VectorXd & state)
{
  Eigen::VectorXd action = getRandomAction();
  Eigen::VectorXd result = getSuccessor(state, action);
  double reward = getReward(state, action, result);
  return Sample(state, action, result, reward);
}

std::vector<Sample> Problem::getRandomTrajectory(const Eigen::VectorXd & initial_state,
                                                 int max_length)
{
  std::vector<Sample> result;
  Eigen::VectorXd state = initial_state;
  while(result.size() < max_length)
  {
    Sample new_sample = getRandomSample(state);
    result.push_back(new_sample);
    if (isTerminal(new_sample.next_state))
      break;
    state = new_sample.next_state;
  }
  return result;
}

std::vector<Sample> Problem::getRandomBatch(const Eigen::VectorXd & initial_state,
                                            int max_length,
                                            int nb_trajectories)
{
  std::vector<Sample> result;
  for (int i = 0; i < nb_trajectories; i++)
  {
    for (const Sample & s : getRandomTrajectory(initial_state, max_length))
    {
      result.push_back(s);
    }
  }
  return result;
}

}
