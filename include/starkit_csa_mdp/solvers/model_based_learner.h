#pragma once

#include "starkit_csa_mdp/solvers/learner.h"

#include "starkit_csa_mdp/reward_predictors/reward_predictor.h"
#include "starkit_csa_mdp/action_optimizers/action_optimizer.h"

#include "starkit_fa/function_approximator.h"
#include "starkit_fa/trainer.h"

#include <random>

namespace csa_mdp
{
/// Experimental algorithm
///
/// Internal config can be learned by
class ModelBasedLearner : public Learner
{
public:
  ModelBasedLearner();

  /// if no policy has been computed yet, return a random policy which
  /// corresponds to the action space
  const std::shared_ptr<const Policy> getPolicy() const;

  //  Problem::RewardFunction getRewardFunction();
  Problem::ValueFunction getValueFunction();

  Eigen::VectorXd getAction(const Eigen::VectorXd& state) override;
  bool hasAvailablePolicy() override;
  void savePolicy(const std::string& prefix) override;
  void saveStatus(const std::string& prefix) override;

  void internalUpdate();

  /// Performs an update of current value using internal parameters
  void updateValue();

  /// Performs an update of current value using internal parameters
  void updatePolicy();

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

private:
  /// The current state of the model
  /// TODO: replace with a pointer of a class which can use provided models and learned models
  std::shared_ptr<const Problem> model;

  /// The horizon finite reward predictor used to create new samples for VFA
  std::unique_ptr<RewardPredictor> reward_predictor;

  /// The value function trainer
  std::unique_ptr<starkit_fa::Trainer> value_trainer;

  /// The current value function
  std::shared_ptr<const starkit_fa::FunctionApproximator> value;

  /// Allows to optimize action using knowledge of the model and knowledge
  /// and of actual policy
  std::unique_ptr<ActionOptimizer> action_optimizer;

  /// The policy trainer
  std::unique_ptr<starkit_fa::Trainer> policy_trainer;

  /// Do we produce stochastic policies? (Should disappear if we use a policy trainer
  /// instead of the fa::Trainer)
  bool use_stochastic_policies;

  /// The current policy
  std::shared_ptr<const Policy> policy;

  /// Random machine
  std::default_random_engine engine;
};

}  // namespace csa_mdp
