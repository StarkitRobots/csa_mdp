#include "rosban_csa_mdp/core/history.h"

#include <fstream>
#include <sstream>

namespace csa_mdp
{

History::History()
{
}

void History::push(const Eigen::VectorXd &state,
                   const Eigen::VectorXd &action,
                   double reward)
{
  states.push_back(state);
  actions.push_back(action);
  rewards.push_back(reward);
}

std::vector<Sample> History::getBatch() const
{
  std::vector<Sample> samples;
  if (states.size() == 0) return samples;
  for (size_t i = 0; i < states.size() - 1; i++)
  {
    samples.push_back(Sample(states[i], actions[i], states[i+1], rewards[i + 1]));
  }
  return samples;
}

static std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while(getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

History History::readCSV(const std::string &path,
                         const std::vector<size_t> &state_cols,
                         const std::vector<size_t> &action_cols,
                         int reward_col,
                         bool header)
{
  History h;
  int x_dim = state_cols.size();
  int u_dim = action_cols.size();
  std::ifstream infile(path);
  std::string line;
  while (std::getline(infile, line))
  {
    // Skipping first line if needed
    if (header)
    {
      header = false;
      continue;
    }
    // Getting columns
    std::vector<std::string> cols = split(line,',');
    // Declaring variables
    Eigen::VectorXd state(x_dim);
    Eigen::VectorXd action(u_dim);
    double reward = 0;
    // Attributing variables at the right place
    int dim = 0;
    for (int col : state_cols){
      state(dim++) = std::stod(cols[col]);
    }
    dim = 0;
    for (int col : action_cols){
      action(dim++) = std::stod(cols[col]);
    }
    if (reward_col >= 0)
      reward = std::stod(cols[reward_col]);
    // Pushing values
    h.push(state, action, reward);
  }
  return h;
}

History History::readCSV(const std::string &path,
                         const std::vector<size_t> &state_cols,
                         const std::vector<size_t> &action_cols,
                         Problem::RewardFunction compute_reward,
                         bool header)
{
  // Factorizing code
  History h = History::readCSV(path, state_cols, action_cols, -1, header);
  // Computing rewards
  for (size_t i = 1; i < h.states.size(); i++)
  {
    h.rewards[i] = compute_reward(h.states[i-1], h.actions[i-1], h.states[i]);
  }
  return h;
}

}
