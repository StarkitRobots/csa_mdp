#pragma once

#include "rosban_csa_mdp/core/black_box_problem.h"
#include "rosban_csa_mdp/core/policy.h"
#include "rosban_csa_mdp/value_approximators/value_approximator.h"

#include "rosban_fa/function_approximator.h"
#include "rosban_fa/optimizer_trainer.h"

#include "rosban_utils/serializable.h"
#include "rosban_utils/time_stamp.h"

#include <memory>

namespace csa_mdp
{

/// Interface for black_box learning algorithms.
/// unlike the 'Learner' objects, 'BlackBoxLearners' are not fed with
/// samples, they interact directly with the blackbox model and can choose
/// any action from any state.
class BlackBoxLearner : public rosban_utils::Serializable {
public:
  BlackBoxLearner();
  ~BlackBoxLearner();

  /// Use the allocated time to find a policy and returns it
  void run(std::default_random_engine * engine);

  /// Update value function based on current policy
  virtual void updateValue(std::default_random_engine * engine) = 0;

  /// Update policy based on current value function
  virtual void updatePolicy(std::default_random_engine * engine) = 0;

  /// Return the average score of the given policy
  virtual double evaluatePolicy(std::default_random_engine * engine);

  /// Set the maximal number of threads allowed
  virtual void setNbThreads(int nb_threads);

  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

protected:
  /// The problem to solve
  std::shared_ptr<const BlackBoxProblem> problem;

  /// The number of threads allowed for the learner
  int nb_threads;

  /// The beginning of the learning process
  rosban_utils::TimeStamp learning_start;

  /// Time allocated for the learning experiment [s]
  double allowed_time;

  /// Discount factor used for the learning process
  double discount;

  /// Number of steps in an evaluation trial
  int trial_length;

  /// Number of evaluation trials to approximate the score of a policy
  int nb_evaluation_trials;

  /// Best expected score reached by a policy at the moment
  double best_score;

  /// Current approximation of the policy function
  std::unique_ptr<Policy> policy;

  /// Current approximation of the value function
  std::unique_ptr<rosban_fa::FunctionApproximator> value;

  /// The approximator used to update the value function
  std::unique_ptr<ValueApproximator> value_approximator;

  /// TODO: make an abstract class to allow different action_optimizers
  /// The optimizer used to train policies
  std::unique_ptr<rosban_fa::OptimizerTrainer> policy_trainer;
};

}
