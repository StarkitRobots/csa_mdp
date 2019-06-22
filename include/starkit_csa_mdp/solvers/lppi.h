#pragma once

#include "starkit_csa_mdp/solvers/black_box_learner.h"
#include "starkit_csa_mdp/online_planners/open_loop_planner.h"

#include "starkit_fa/trainer.h"

namespace starkit_fa
{
class FunctionApproximator;
}

namespace csa_mdp
{
class LPPI : public BlackBoxLearner
{
public:
  LPPI();
  virtual ~LPPI();

  /// Perform a single rollout and stores the results
  /// - states  : each column will be a different state
  /// - actions : each column will be a different action
  /// - values  : each row is the sampled value for the state
  void performRollout(Eigen::MatrixXd* states, Eigen::MatrixXd* actions, Eigen::VectorXd* values,
                      std::default_random_engine* engine);

  /// Perform the rollouts and store the results
  /// - states  : each column will be a different state
  /// - actions : each column will be a different action
  /// - values  : each row is the sampled value for a state
  void performRollouts(Eigen::MatrixXd* states, Eigen::MatrixXd* actions, Eigen::VectorXd* values,
                       std::default_random_engine* engine);

  virtual void init(std::default_random_engine* engine) override;

  /// Perform rollouts according to the OpenLoopPlanner
  virtual void update(std::default_random_engine* engine) override;

  void updateValues(const Eigen::MatrixXd& states, const Eigen::VectorXd& values);
  /// Return the new policy_fa, but do not modify object
  std::unique_ptr<starkit_fa::FunctionApproximator> updatePolicy(const Eigen::MatrixXd& states,
                                                                const Eigen::MatrixXd& actions) const;

  virtual void setNbThreads(int nb_threads) override;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

private:
  /// LPPI uses an open loop planner for optimization of actions
  OpenLoopPlanner planner;
  /// A forest describing current value function
  std::unique_ptr<starkit_fa::FunctionApproximator> value;
  /// A forest describing the value function trainer
  std::unique_ptr<starkit_fa::Trainer> value_trainer;
  /// The current policy
  std::unique_ptr<Policy> policy;
  /// The function approximator used as a basis for the current policy
  std::unique_ptr<starkit_fa::FunctionApproximator> policy_fa;
  /// A forest describing policy trainer
  std::unique_ptr<starkit_fa::Trainer> policy_trainer;
  /// Minimal remaining length of a rollout to allow use of an entry if the
  /// rollout does not end with a terminal status
  int min_rollout_length;
  /// Maximal length for a rollout
  int max_rollout_length;
  /// Number of entries per update
  int nb_entries;
  /// The best average reward for policies encountered:
  /// - It is used to choose if we update the policy
  double best_reward;
  /// If true, current policy is used to finish the trials, otherwise, value
  /// estimation is used to estimate remaining value
  bool use_policy;
};

}  // namespace csa_mdp
