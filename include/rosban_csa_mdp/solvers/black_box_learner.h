#pragma once

#include "rosban_csa_mdp/core/black_box_problem.h"
#include "rosban_csa_mdp/core/policy.h"
#include "rosban_csa_mdp/value_approximators/value_approximator.h"

#include "rosban_fa/function_approximator.h"
#include "rosban_fa/optimizer_trainer.h"

#include "rosban_utils/serializable.h"
#include "rosban_utils/time_stamp.h"

#include <fstream>
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
  virtual void updateValue(std::default_random_engine * engine);

  /// Return a new policy based on current value function
  virtual std::unique_ptr<Policy> updatePolicy(std::default_random_engine * engine);

  /// Return the average score of the given policy
  virtual double evaluatePolicy(const Policy & p,
                                std::default_random_engine * engine);

  /// Set the maximal number of threads allowed
  virtual void setNbThreads(int nb_threads);

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

  /// Open all logs streams
  void openLogs();

  /// Close all logs streams
  void closeLogs();

  /// Dump the time consumption to the time file
  void writeTime(const std::string & name, double time);

  /// Dump the score at current iteration
  void writeScore(double score);

protected:
  /// The problem to solve
  std::shared_ptr<const BlackBoxProblem> problem;

  /// The number of threads allowed for the learner
  int nb_threads;

  /// The beginning of the learning process
  rosban_utils::TimeStamp learning_start;

  /// Time allocated for the learning experiment [s]
  double time_budget;

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

  /// If enabled, then memory of the policy_trainer is emptied before each
  /// training process
  bool memoryless_policy_trainer;

  /// Does the learner use an approximation of the value
  bool use_value_approximator;

  /// Number of iterations performed
  int iterations;

  /// Storing time logs
  std::ofstream time_file;

  /// Storing evaluation logs
  std::ofstream results_file;
};

}
