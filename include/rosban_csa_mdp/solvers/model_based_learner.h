#pragma once

#include "rosban_csa_mdp/core/model.h"
#include "rosban_csa_mdp/reward_predictors/reward_predictor.h"
#include "rosban_csa_mdp/action_optimizers/action_optimizer.h"

#include "rosban_fa/function_approximator.h"

namespace csa_mdp
{

/// Experimental algorithm
///
/// Internal config can be learned by 
class ModelBasedLearner : public rosban_utils::Serializable
{
public:

  //TODO!!!

  void internalUpdate();

  /// Performs an update of current value using internal parameters
  void updateValue();

private:
  /// The current state of the model
  /// TODO: replace with a pointer of a class which can use provided models and learned models
  std::shared_ptr<Problem> model;


  /// The horizon finite reward predictor used to create new samples for VFA
  std::unique_ptr<RewardPredictor> reward_predictor;

  /// The value function trainer
  std::unique_ptr<rosban_fa::Trainer> value_trainer;

  /// The current value function
  std::shared_ptr<const rosban_fa::FunctionApproximator> value;

  /// Allows to optimize action using knowledge of the model and knowledge
  /// and of actual policy
  std::unique_ptr<ActionOptimizer> action_optimizer;

  /// The policy trainer
  std::unique_ptr<const rosban_fa::Trainer> policy_trainer;

  /// The current policy
  std::shared_ptr<const Policy> policy;

  /// Acquired samples until now
  std::vector<Sample> samples;
};
