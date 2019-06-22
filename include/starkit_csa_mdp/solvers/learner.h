#pragma once

#include "starkit_csa_mdp/core/problem.h"

#include "starkit_utils/serialization/json_serializable.h"
#include "starkit_utils/timing/time_stamp.h"

namespace csa_mdp
{
/// Interface for learning algorithms
class Learner : public starkit_utils::JsonSerializable
{
public:
  Learner();
  virtual ~Learner();

  /// Inform the learner that the process just started
  void setStart();

  /// Set the maximal number of threads allowed to the learner
  virtual void setNbThreads(int nb_threads);

  /// Inform the learner that the current run is finished
  virtual void endRun();

  /// Inform the learner of the state space
  virtual void setStateLimits(const Eigen::MatrixXd& state_limits);
  /// Inform the learner of the action space
  virtual void setActionLimits(const std::vector<Eigen::MatrixXd>& action_limits);
  /// Update the discount factor used for the learner
  virtual void setDiscount(double discount);

  Eigen::MatrixXd getStateLimits() const;
  std::vector<Eigen::MatrixXd> getActionLimits() const;

  /// Return the action which needs to be taken in the given state
  virtual Eigen::VectorXd getAction(const Eigen::VectorXd& state) = 0;

  // TODO incorporate the 'terminal action' flag?
  /// Add the sample to the sample collection and take any required action
  virtual void feed(const csa_mdp::Sample& sample);

  /// Update the content of the learner, this method should not be called during
  /// a trial because it is likely to require a lot of time
  virtual void internalUpdate() = 0;

  // TODO might be removed if RandomPolicy can be saved
  /// Has a policy already been computed by the learner ?
  virtual bool hasAvailablePolicy() = 0;

  /// Save current policy with the given prefix
  virtual void savePolicy(const std::string& prefix) = 0;

  /// Save current status with the given prefix
  virtual void saveStatus(const std::string& prefix) = 0;

  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

  /// Return time elapsed since learning start (in seconds)
  double getLearningTime() const;

  /// Retrieve the time repartition of the last internal update
  /// Entries have the following format: <label, time[s]>
  const std::map<std::string, double>& getTimeRepartition() const;

protected:
  /// Acquired samples until now
  std::vector<Sample> samples;

protected:
  /// This function allows to test if a state is terminal
  std::function<bool(const Eigen::VectorXd&)> terminal_function;
  /// The discount factor of the learning process
  double discount;
  /// The number of threads allowed to the learner
  int nb_threads;
  /// The beginning of the learning
  starkit_utils::TimeStamp learning_start;
  /// Store time repartition for the last internal update [s]
  std::map<std::string, double> time_repartition;

private:
  /// Limits of state space used for learning of:
  /// - model dynamics
  /// - value function
  /// - policy
  Eigen::MatrixXd state_limits;
  /// Limits of the action space
  std::vector<Eigen::MatrixXd> action_limits;
};

}  // namespace csa_mdp
