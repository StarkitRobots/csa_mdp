#pragma once

#include "rosban_csa_mdp/core/sample.h"

#include "rosban_regression_forests/core/training_set.h"
#include "rosban_regression_forests/algorithms/extra_trees.h"

namespace csa_mdp
{

/// FPF acronym stands for Fitted Policy Forest. This algorithms (unpublished yet) is based
/// on FQI (cf. Tree-Based Batch Mode Reinforcement Learning. Ernst, Geurts & Wehenkel, 2005).
/// The main additions over FQI are:
/// - The choice of the best action for a given state by merging a regression forest into a
///   single tree. 
/// - The possibility to learn a fitted policy from the q-value
class FPF {
public:
  class Config : public rosban_utils::Serializable{
  private:
    // Storing x_dim and u_dim is required in order to load properly a configuration
    // TODO: not exactly in fact, dividing size of read vector by 2 should give the
    //       expected result

    /// The number of continuous dimensions for the state
    int x_dim;
    /// The number of continuous dimensions for actions
    int u_dim;
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
    size_t max_action_tiles;
    /// > 0: The number of samples generated to learn the policy
    /// = 0: States used to learn policy are the same as the state from samples
    /// < 0: Policy is not learned
    int policy_samples;
    /// The time spent learning the q_value [s]
    double q_value_time;
    /// The time spent learning the policy from the q_value [s]
    double policy_time;
    /// Number of threads used to compute the trainingset
    int nb_threads;

    /// Config used for computing the Q-value
    regression_forests::ExtraTrees::Config q_value_conf;
    /// Config used for computing the Policy
    regression_forests::ExtraTrees::Config policy_conf;

    Config();

    const Eigen::MatrixXd & getStateLimits() const;
    const Eigen::MatrixXd & getActionLimits() const;
    /// Return the space limits for the input State, then action
    Eigen::MatrixXd getInputLimits() const;

    void setStateLimits(const Eigen::MatrixXd &new_limits);
    void setActionLimits(const Eigen::MatrixXd &new_limits);

    // XML stuff
    virtual std::string class_name() const override;
    virtual void to_xml(std::ostream &out) const override;
    virtual void from_xml(TiXmlNode *node) override;
  };

protected:
  /// A forest describing current q_value
  std::unique_ptr<regression_forests::Forest> q_value;
  /// Since action might be multi-dimensional, it is necessary to represent the
  /// policy by one forest for each dimension. This choice might lead to unsatisfying
  /// results, depending on the shape of the quality function with respect to the action
  std::vector<std::unique_ptr<regression_forests::Forest>> policies;

  /// Create a TrainingSet from current q_value and a collection f mdp samples
  /// Note: this method is virtual, because other algorithms (such as MRE) might need to use a
  ///       custom way of creating the trainingSet
  virtual regression_forests::TrainingSet
  getTrainingSet(const std::vector<Sample>& samples,
                 std::function<bool(const Eigen::VectorXd&)> is_terminal,
                 const Config &conf);

  /// Create a TrainingSet from current q_value, using samples from samples[start_idx,end_idx[
  virtual regression_forests::TrainingSet
  getTrainingSet(const std::vector<Sample>& samples,
                 std::function<bool(const Eigen::VectorXd&)> is_terminal,
                 const Config &conf,
                 int start_idx, int end_idx);

  /// Perform one step of update on the Q-value. This function is virtual because some
  /// algorithms need to modify it
  virtual void updateQValue(const std::vector<Sample>& samples,
                            std::function<bool(const Eigen::VectorXd&)> isTerminal,
                            const Config &conf);

  /// Create a set of states which will be used to build the the policy forest, in the default implementation,
  /// states are chosen at uniformous random inside the state space
  /// Note: this method is virtual, because other algorithms (such as MRE) might need to use a
  ///       custom way of creating their policy training state
  virtual std::vector<Eigen::VectorXd>
  getPolicyTrainingStates(const std::vector<Sample>& samples,
                          const Config &conf);

  /// Compute the bestAction at given state according to the current q_value
  Eigen::VectorXd bestAction(const Eigen::VectorXd& state);

public:
  /// Create a FPF solver with a default configuration
  FPF();

  const regression_forests::Forest& getValueForest();
  const regression_forests::Forest& getPolicyForest(int action_index);

  /// Remove the forest from the memory of the solver!!!
  std::unique_ptr<regression_forests::Forest> stealPolicyForest(int action_index);

  void solve(const std::vector<Sample>& samples,
             std::function<bool(const Eigen::VectorXd&)> is_terminal,
             Config &conf);
};


}
