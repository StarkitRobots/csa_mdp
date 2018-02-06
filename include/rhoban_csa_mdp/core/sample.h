#pragma once

#include <Eigen/Core>

#include <vector>

namespace csa_mdp
{

/// This class wraps a 4-tuple (s,a, s', r) which corresponds to a sample of a continuous state mdp
class Sample
{
public:
  Eigen::VectorXd state;
  Eigen::VectorXd action;
  Eigen::VectorXd next_state;
  double reward;
  
  Sample();
  Sample(const Eigen::VectorXd &state,
         const Eigen::VectorXd &action,
         const Eigen::VectorXd &next_state,
         double reward);

  /// Read a set of Samples from a csv file, user needs to specify the index of the columns
  static std::vector<Sample> readCSV(const std::string &path,
                                     const std::vector<size_t> &src_state_cols,
                                     const std::vector<size_t> &action_cols,
                                     const std::vector<size_t> &dst_state_cols,
                                     int reward_col,
                                     bool header);
};

}
