#include "rhoban_csa_mdp/core/sample.h"

#include <fstream>

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

//TODO externalize to another module
static std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while(getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

std::vector<Sample> Sample::readCSV(const std::string &path,
                                    const std::vector<size_t> &src_state_cols,
                                    const std::vector<size_t> &action_cols,
                                    const std::vector<size_t> &dst_state_cols,
                                    int reward_col,
                                    bool header)
{
  std::vector<Sample> samples;
  int x_dim = src_state_cols.size();
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
    Eigen::VectorXd src_state(x_dim);
    Eigen::VectorXd action(u_dim);
    Eigen::VectorXd dst_state(x_dim);
    double reward = 0;
    // Attributing variables at the right place
    int dim = 0;
    for (int col : src_state_cols){
      src_state(dim++) = std::stod(cols[col]);
    }
    dim = 0;
    for (int col : action_cols){
      action(dim++) = std::stod(cols[col]);
    }
    dim = 0;
    for (int col : dst_state_cols){
      dst_state(dim++) = std::stod(cols[col]);
    }
    if (reward_col >= 0)
      reward = std::stod(cols[reward_col]);
    // Pushing values
    samples.push_back(Sample(src_state, action, dst_state, reward));
  }
  return samples;
}


}
