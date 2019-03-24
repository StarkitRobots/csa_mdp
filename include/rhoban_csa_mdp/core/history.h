#pragma once

#include "rhoban_csa_mdp/core/problem.h"

#include <memory>

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
  /// Describe the config used to load an History or a vector of History
  class Config : public rhoban_utils::JsonSerializable
  {
  public:
    Config();

    std::string getClassName() const override;
    virtual Json::Value toJson() const override;
    virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

    std::string log_path;
    int run_column;
    int step_column;
    std::vector<int> state_columns;
    std::vector<int> action_columns;
    std::shared_ptr<Problem> problem;
  };

  History();

  void push(const Eigen::VectorXd& state, const Eigen::VectorXd& action, double reward);

  /// Transform the content of the history to a batch format (s,a,s',r)
  std::vector<Sample> getBatch() const;
  /// Transform the content of several histories
  static std::vector<Sample> getBatch(const std::vector<History>& histories);

  static std::vector<History> readCSV(const History::Config& conf);

  /// Read the history contained in a csv file with classical form:
  /// - column 0 is 'run'
  /// - column 1 is 'step'
  /// - the next 'nb_states' column are the state values
  /// - the next 'nb_actions' column are the action values
  /// - the last column is the reward
  /// The file also need to have a header
  static std::vector<History> readCSV(const std::string& path, int nb_states, int nb_actions);

  /// Read the history contained in a csv file
  /// - path: the location of the file
  /// - run_col: this column indicate the id of the run
  /// - step_col: this column indicate the id of the step
  /// - state_cols: the columns containing the state dimensions
  /// - action_cols: the columns containing the action dimensions
  /// - reward_col: the column containing the rewards (if reward_col < 0, reward is not read)
  /// - header: Does the file contain a header?
  static std::vector<History> readCSV(const std::string& path, int run_col, int step_col,
                                      const std::vector<int>& state_cols, const std::vector<int>& action_cols,
                                      int reward_col, bool header);

  /// Read the history contained in a csv file
  /// - path: the location of the file
  /// - run_col: this column indicate the id of the run
  /// - step_col: this column indicate the id of the step
  /// - state_cols: the columns containing the state dimensions
  /// - action_cols: the columns containing the action dimensions
  /// - compute_reward: a function allowing to get the reward from (s,a,s')
  /// - header: Does the file contain a header?
  static std::vector<History> readCSV(const std::string& path, int run_col, int step_col,
                                      const std::vector<int>& state_cols, const std::vector<int>& action_cols,
                                      Problem::RewardFunction compute_reward, bool header);

  /// Read the history contained in a csv file
  /// - path: the location of the file
  /// - state_cols: the columns containing the state dimensions
  /// - action_cols: the columns containing the action dimensions
  /// - reward_col: the column containing the rewards (if reward_col < 0, reward is not read)
  /// - header: Does the file contain a header?
  static History readCSV(const std::string& path, const std::vector<int>& state_cols,
                         const std::vector<int>& action_cols, int reward_col, bool header);

  /// Read the history contained in a csv file
  /// - path: the location of the file
  /// - state_cols: the columns containing the state dimensions
  /// - action_cols: the columns containing the action dimensions
  /// - compute_reward: a function allowing to get the reward from (s,a,s')
  /// - header: Does the file contain a header?
  static History readCSV(const std::string& path, const std::vector<int>& state_cols,
                         const std::vector<int>& action_cols, Problem::RewardFunction compute_reward, bool header);
};

}  // namespace csa_mdp
