#pragma once

#include "problem.h"

namespace csa_mdp
{

/// This class allows to represent informations received from a single run of a problem.
/// The information stored in this class can mainly be described as 3-tuples of the form:
/// ((s(t) a(t) r(t)), (s(t+1) a(t+1) r(t+1)), ...)
///
/// Informations must be pushed in the appropriate order
class History
{
private:
  std::vector<Eigen::VectorXd> states;
  std::vector<Eigen::VectorXd> actions;
  std::vector<double> rewards;

public:
  History();

  void push(const Eigen::VectorXd &state,
            const Eigen::VectorXd &action,
            double reward);

  /// Transform the content of the history to a batch format (s,a,s',r)
  std::vector<Sample> getBatch() const;

  /// Read the history contained in a csv file
  /// - path: the location of the file
  /// - state_cols: the columns containing the state dimensions
  /// - action_cols: the columns containing the action dimensions
  /// - reward_col: the column containing the rewards (if reward_col < 0, reward is not read)
  /// - header: Does the file contain a header?
  static History readCSV(const std::string &path,
                         const std::vector<size_t> &state_cols,
                         const std::vector<size_t> &action_cols,
                         int reward_col,
                         bool header);

  /// Read the history contained in a csv file
  /// - path: the location of the file
  /// - state_cols: the columns containing the state dimensions
  /// - action_cols: the columns containing the action dimensions
  /// - compute_reward: a function allowing to get the reward from (s,a,s')
  /// - header: Does the file contain a header?
  static History readCSV(const std::string &path,
                         const std::vector<size_t> &state_cols,
                         const std::vector<size_t> &action_cols,
                         Problem::RewardFunction compute_reward,
                         bool header);
};

}
