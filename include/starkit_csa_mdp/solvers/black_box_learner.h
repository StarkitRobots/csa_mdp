#pragma once

#include "starkit_csa_mdp/core/black_box_problem.h"
#include "starkit_csa_mdp/core/policy.h"
#include "starkit_csa_mdp/value_approximators/value_approximator.h"

#include "starkit_fa/function_approximator.h"
#include "starkit_fa/optimizer_trainer.h"

#include "starkit_utils/serialization/json_serializable.h"
#include "starkit_utils/timing/time_stamp.h"

#include <fstream>
#include <memory>

namespace csa_mdp
{
/// Interface for black_box learning algorithms.
/// unlike the 'Learner' objects, 'BlackBoxLearners' are not fed with
/// samples, they interact directly with the blackbox model and can choose
/// any action from any state.
class BlackBoxLearner : public starkit_utils::JsonSerializable
{
public:
  BlackBoxLearner();
  virtual ~BlackBoxLearner();

  /// Build a policy from the given function approximator
  std::unique_ptr<Policy> buildPolicy(const starkit_fa::FunctionApproximator& fa) const;

  // Initialize the learner
  virtual void init(std::default_random_engine* engine) = 0;

  /// Use the allocated time to find a policy and returns it
  void run(std::default_random_engine* engine);

  /// Perform a single step of update of an iterative learner
  virtual void update(std::default_random_engine* engine) = 0;

  /// Use nb_evaluation_trials evaluations
  virtual double evaluatePolicy(const Policy& p, std::default_random_engine* engine) const;

  /// Return the average score of the given policy using 'nb_evaluations' trajectories
  /// If nb_evaluations is not a nullptr, then add all the visited states to the provided
  /// vector
  virtual double evaluatePolicy(const Policy& p, int nb_evaluations, std::default_random_engine* engine,
                                std::vector<Eigen::VectorXd>* visited_states = nullptr) const;

  /// Evaluate the average reward for policy p, for an uniform distribution in
  /// space, using nb_evaluations trials.
  double localEvaluation(const Policy& p, const Eigen::MatrixXd& space, int nb_evaluations,
                         std::default_random_engine* engine) const;

  /// Evaluate the policy for a set of given initial states
  double evaluation(const Policy& p, const std::vector<Eigen::VectorXd>& initial_states,
                    std::default_random_engine* engine) const;

  /// Set the maximal number of threads allowed
  virtual void setNbThreads(int nb_threads);

  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

  /// Open all logs streams
  void openLogs();

  /// Close all logs streams
  void closeLogs();

  /// Dump the time consumption to the time file
  void writeTime(const std::string& name, double time);

  /// Dump the score at current iteration
  void writeScore(double score);

protected:
  /// The problem to solve
  std::shared_ptr<const BlackBoxProblem> problem;

  /// The number of threads allowed for the learner
  int nb_threads;

  /// The beginning of the learning process
  starkit_utils::TimeStamp learning_start;

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

}  // namespace csa_mdp
