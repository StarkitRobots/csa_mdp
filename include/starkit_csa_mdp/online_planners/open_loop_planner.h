#pragma once

#include "starkit_csa_mdp/core/problem.h"

#include "starkit_utils/serialization/json_serializable.h"
#include "starkit_fa/function_approximator.h"
#include "starkit_bbo/optimizer.h"

namespace csa_mdp
{
class OpenLoopPlanner : public starkit_utils::JsonSerializable
{
public:
  OpenLoopPlanner();

  /// Check that the OpenLoopPlanner is properly configured for the given problem
  void checkConsistency(const Problem& p) const;

  /// Configure the optimizer for the given problem
  void prepareOptimizer(const Problem& p);

  /// Sample the reward received by applying all the 'next_actions' from
  /// 'initial_state' with problem 'p'. The last state of the rollout is stored
  /// in 'last_state' and if a terminal status is received, 'is_terminated' is
  /// set to true.
  double sampleLookAheadReward(const Problem& p, const Eigen::VectorXd& initial_state,
                               const Eigen::VectorXd& next_actions, Eigen::VectorXd* last_state, bool* is_terminated,
                               std::default_random_engine* engine);

  /// Optimize the next action for the given problem 'p' starting at 'state'
  /// according to inner parameters and provided value_function
  Eigen::VectorXd planNextAction(const Problem& p, const Eigen::VectorXd& state, const Policy& policy,
                                 std::default_random_engine* engine);

  /// Optimize the next action for the given problem 'p' starting at 'state'
  /// according to inner parameters and provided value_function
  Eigen::VectorXd planNextAction(const Problem& p, const Eigen::VectorXd& state,
                                 const starkit_fa::FunctionApproximator& value_function,
                                 std::default_random_engine* engine);

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

private:
  /// Optimizer used for open loop planning
  std::unique_ptr<starkit_bbo::Optimizer> optimizer;

  /// Number of steps of look ahead
  int look_ahead;

  /// Number of steps for a trial when using a default policy after step
  int trial_length;

  /// Number of rollouts used to average the reward at each sample
  int rollouts_per_sample;

  /// Discount value used for optimization
  double discount;
};

}  // namespace csa_mdp
