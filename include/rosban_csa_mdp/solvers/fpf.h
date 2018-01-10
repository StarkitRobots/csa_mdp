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
  class Config : public rhoban_utils::JsonSerializable{
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
    /// The time spent computing training sets for q_value [s]
    double q_training_set_time;
    /// The time spent growing extra-trees for the q_value [s]
    double q_extra_trees_time;
    /// The time spent computing training sets for policies [s]
    double p_training_set_time;
    /// The time spent growing extra-trees for the policies [s]
    double p_extra_trees_time;
    /// Number of threads used to compute the trainingset
    int nb_threads;
    /// If activated, internal config are ignored and replaced by heuristic based
    /// parameters. (see ExtraTrees::Config::generateAuto)
    bool auto_parameters;
    /// If activated, gaussian processes are used to represent the values
    /// and max is computed according to gradient
    bool gp_values;
    /// If activated, gaussian processes are used to represent the policies
    bool gp_policies;

    /// Config used for computing the Q-value
    regression_forests::ExtraTrees::Config q_value_conf;
    /// Config used for computing the Policy
    regression_forests::ExtraTrees::Config policy_conf;

    /// Config used for auto_tuning of GP hyperparameters
    rosban_gp::RandomizedRProp::Config hyper_rprop_conf;
    /// Config used for gradient ascent when using gp
    rosban_gp::RandomizedRProp::Config find_max_rprop_conf;

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

protected:
  /// A forest describing current q_value
  std::unique_ptr<regression_forests::Forest> q_value;
  /// Since action might be multi-dimensional, it is necessary to represent the
  /// policy by one forest for each dimension. This choice might lead to unsatisfying
  /// results, depending on the shape of the quality function with respect to the action
  std::vector<std::unique_ptr<regression_forests::Forest>> policies;

  /// Create a TrainingSet from current q_value and a collection f mdp samples
  /// Note: this method is not virtual, because if other algorithms (such as MRE)
  ///       need to use a custom way of initializing the data, they should implement
  ///       the method which generate samples for a given interval
  regression_forests::TrainingSet
  getTrainingSet(const std::vector<Sample>& samples,
                 std::function<bool(const Eigen::VectorXd&)> is_terminal,
                 const Config &conf);

  /// Create a TrainingSet from current q_value, using samples from samples[start_idx,end_idx[
  virtual regression_forests::TrainingSet
  getTrainingSet(const std::vector<Sample>& samples,
                 std::function<bool(const Eigen::VectorXd&)> is_terminal,
                 const Config &conf,
                 int start_idx, int end_idx);

  /// Perform one step of update on the Q-value, last_step might include special update.
  /// This function is virtual because some algorithms need to modify it.
  virtual void updateQValue(const std::vector<Sample>& samples,
                            std::function<bool(const Eigen::VectorXd&)> isTerminal,
                            Config &conf,
                            bool last_step);

  /// Create a set of states which will be used to build the the policy forest, in the default implementation,
  /// states are chosen at uniformous random inside the state space
  /// Note: this method is virtual, because other algorithms (such as MRE) might need to use a
  ///       custom way of creating their policy training state
  virtual std::vector<Eigen::VectorXd>
  getPolicyTrainingStates(const std::vector<Sample>& samples,
                          const Config &conf);

  /// This function is not virtual, because it mainly handle the multi threading
  std::vector<Eigen::VectorXd>
  getPolicyActions(const std::vector<Eigen::VectorXd> &states,
                   const Config &conf);

  virtual std::vector<Eigen::VectorXd>
  getPolicyActions(const std::vector<Eigen::VectorXd> &states,
                   const Config &conf,
                   int start_idx, int end_idx);

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
