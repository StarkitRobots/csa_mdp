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
  virtual ~BlackBoxLearner();

  // Initialize the learner
  virtual void init(std::default_random_engine * engine) = 0;

  /// Use the allocated time to find a policy and returns it
  void run(std::default_random_engine * engine);

  /// Perform a single step of update of an iterative learner
  virtual void update(std::default_random_engine * engine) = 0;

  /// Return the average score of the given policy, using a state chosen
  /// randomly according to problem
  virtual double evaluatePolicy(const Policy & p,
                                std::default_random_engine * engine) const;

  /// Evaluate the average reward for policy p, for an uniform distribution in
  /// space, using nb_evaluations trials.
  double localEvaluation(const Policy & p,
                         const Eigen::MatrixXd & space,
                         int nb_evaluations,
                         std::default_random_engine * engine) const;

  /// Set the maximal number of threads allowed
  virtual void setNbThreads(int nb_threads);

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

  /// Number of iterations performed
  int iterations;

  /// Verbosity level of the learner
  int verbosity;

  /// Storing time logs
  std::ofstream time_file;

  /// Storing evaluation logs
  std::ofstream results_file;
};

}
