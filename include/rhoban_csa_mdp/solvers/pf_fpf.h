#pragma once

#include "rhoban_csa_mdp/core/sample.h"
#include "rhoban_csa_mdp/knownness/knownness_function.h"

#include "rhoban_regression_forests/core/training_set.h"
#include "rhoban_regression_forests/algorithms/extra_trees.h"

namespace csa_mdp
{

/// This class implements a Parameter-Free Fitted Policy Forest
/// - It is inspired from Fitted Policy Forest and uses notion of knownness
class PF_FPF
{
public:
  /// The configuration include all the necessary information for computing 
  class Config : public rhoban_utils::JsonSerializable{
  private:
    /// The number of continuous dimensions for the state
    size_t x_dim;
    /// The number of continuous dimensions for actions
    size_t u_dim;
    /// The state space
    Eigen::MatrixXd x_limits;
    /// The action space
    Eigen::MatrixXd u_limits;

  public:
    /// Until which horizon should value be computed
    size_t horizon;
    /// The discount factor of the MDP
    double discount;
    /// The final size of the tree when merging the forest into a single tree for a given action
    /// TODO: this parameter should be set automatically, but how?
    size_t max_action_tiles;
    /// The time spent learning the q_value [s]
    double q_value_time;
    /// The time spent learning the policy from the q_value [s]
    double policy_time;
    /// Number of threads used to compute
    int nb_threads;
    /// The maximal reward reachable
    double reward_max;

    Config();

    const Eigen::MatrixXd & getStateLimits() const;
    const Eigen::MatrixXd & getActionLimits() const;
    /// Return the space limits for the input State, then action
    Eigen::MatrixXd getInputLimits() const;

    void setStateLimits(const Eigen::MatrixXd &new_limits);
    void setActionLimits(const Eigen::MatrixXd &new_limits);

    // XML stuff
    virtual std::string getClassName() const override;
    virtual Json::Value toJson() const override;
    virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;
  };

private:

  PF_FPF();

  /// Create a TrainingSet from given q_value and a collection of mdp samples
  regression_forests::TrainingSet
  getTrainingSet(const std::vector<Sample> &samples,
                 std::function<bool(const Eigen::VectorXd&)> is_terminal,
                 const Config &conf,
                 std::shared_ptr<regression_forests::Forest> q_value);

  /// Create a TrainingSet from given q_value, using samples from samples[start_idx,end_idx[
  /// This function makes multi-threading easier
  regression_forests::TrainingSet
  getTrainingSet(const std::vector<Sample>& samples,
                 std::function<bool(const Eigen::VectorXd&)> is_terminal,
                 const Config &conf,
                 std::shared_ptr<regression_forests::Forest> q_value,
                 int start_idx, int end_idx);

  /// Perform one step of update on the Q-value.
  std::unique_ptr<regression_forests::Forest>
  updateQValue(const std::vector<Sample>& samples,
               std::function<bool(const Eigen::VectorXd&)> isTerminal,
               const Config &conf,
               std::unique_ptr<regression_forests::Forest> q_value,
               bool final_step);
public:

  /// Compute the q-value according to the provided configuration
  std::unique_ptr<regression_forests::Forest>
  generateQValue(const std::vector<Sample>& samples,
                 std::function<bool(const Eigen::VectorXd&)> is_terminal,
                 Config &conf);

  std::vector<std::unique_ptr<regression_forests::Forest>>
  generatePolicy(const std::vector<Sample>& samples,
                 std::unique_ptr<regression_forests::Forest> q_value);

  std::vector<std::vector<std::unique_ptr<regression_forests::Forest>>>
  generatePolicies();

};

}
