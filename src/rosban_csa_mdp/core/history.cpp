#include "rosban_csa_mdp/core/history.h"

#include "rosban_utils/string_tools.h"

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

std::vector<Sample> History::getBatch(const std::vector<History> &histories)
{
  std::vector<Sample> samples;
  for (const History &h : histories)
  {
    std::vector<Sample> new_samples = h.getBatch();
    samples.reserve(samples.size() + new_samples.size());
    std::move(new_samples.begin(), new_samples.end(), std::inserter(samples, samples.end()));
  }
  return samples;
}

std::vector<History> History::readCSV(const std::string &path,
                                      int run_col,
                                      int step_col,
                                      const std::vector<int> &state_cols,
                                      const std::vector<int> &action_cols,
                                      int reward_col,
                                      bool header)
{
  std::vector<History> histories;
  int x_dim = state_cols.size();
  int u_dim = action_cols.size();
  std::ifstream infile(path);
  if (!infile.good()) throw std::runtime_error("Failed to open file '" + path + "'");
  std::string line;
  // Temporary variables
  History curr_history;
  int curr_run = 1;
  int expected_step = 0;
  while (std::getline(infile, line))
  {
    // Skipping first line if needed
    if (header)
    {
      header = false;
      continue;
    }
    // Getting columns
    std::vector<std::string> cols = rosban_utils::split_string(line,',');
    // Checking if a new run is started (if run-col is specified)
    if (run_col >= 0 && std::stoi(cols[run_col]) != curr_run)
    {
      histories.push_back(curr_history);
      curr_run = std::stoi(cols[run_col]);
      histories.clear();
      expected_step = 0;
    }
    // Checking
    if (step_col >= 0 && std::stoi(cols[step_col]) != expected_step)
    {
      std::ostringstream oss;
      oss << "Unexpected step: '" << cols[step_col] << "' expecting step '"
          << expected_step << "'";
      throw std::runtime_error(oss.str());
    }

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
    curr_history.push(state, action, reward);
    expected_step++;
  }
  histories.push_back(curr_history);
  return histories;
}

std::vector<History> History::readCSV(const std::string &path,
                                      int run_col,
                                      int step_col,
                                      const std::vector<int> &state_cols,
                                      const std::vector<int> &action_cols,
                                      Problem::RewardFunction compute_reward,
                                      bool header)
{
  // Factorizing code
  std::vector<History> histories = History::readCSV(path, run_col, step_col,
                                                    state_cols, action_cols, -1, header);
  // Computing rewards
  for (auto & h : histories)
  {
    for (size_t i = 1; i < h.states.size(); i++)
    {
      h.rewards[i] = compute_reward(h.states[i-1], h.actions[i-1], h.states[i]);
    }
  }
  return histories;
}

History History::readCSV(const std::string &path,
                         const std::vector<int> &state_cols,
                         const std::vector<int> &action_cols,
                         int reward_col,
                         bool header)
{
  return readCSV(path, -1, -1, state_cols, action_cols, reward_col, header)[0];
}

History History::readCSV(const std::string &path,
                         const std::vector<int> &state_cols,
                         const std::vector<int> &action_cols,
                         Problem::RewardFunction compute_reward,
                         bool header)
{
  return readCSV(path, -1, -1, state_cols, action_cols, compute_reward, header)[0];
}

}
