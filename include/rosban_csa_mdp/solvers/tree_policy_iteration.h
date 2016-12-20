#pragma once

#include "rosban_csa_mdp/solvers/black_box_learner.h"

namespace csa_mdp
{

class TreePolicyIteration : public BlackBoxLearner {
public:
  TreePolicyIteration();
  virtual ~TreePolicyIteration();

  virtual void init(std::default_random_engine * engine) override;
  virtual void update(std::default_random_engine * engine) override;

  virtual void setNbThreads(int nb_threads) override;

  /// Update value function based on current policy
  void updateValue(std::default_random_engine * engine);

  /// Return a new policy based on current value function
  std::unique_ptr<Policy> updatePolicy(std::default_random_engine * engine);


  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

protected:

  /// Best expected score reached by a policy at the moment
  double best_score;

  /// Current approximation of the policy function
  std::unique_ptr<Policy> policy;

  /// Current approximation of the value function
  std::unique_ptr<rosban_fa::FunctionApproximator> value;

  /// The approximator used to update the value function
  std::unique_ptr<ValueApproximator> value_approximator;

  /// The optimizer used to train policies
  std::unique_ptr<rosban_fa::OptimizerTrainer> policy_trainer;

  /// If enabled, then memory of the policy_trainer is emptied before each
  /// training process
  bool memoryless_policy_trainer;

  /// Does the learner use an approximation of the value
  bool use_value_approximator;

};

}
