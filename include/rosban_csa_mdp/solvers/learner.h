#pragma once

#include "rosban_csa_mdp/core/problem.h"

#include "rosban_utils/serializable.h"

namespace csa_mdp
{

/// Interface for learning algorithms
class Learner : public rosban_utils::Serializable
{
public:
  /// Inform the learner of the state space
  virtual void setStateLimits(const Eigen::MatrixXd & state_limits);
  /// Inform the learner of the action space
  virtual void setActionLimits(const Eigen::MatrixXd & action_limits);

  Eigen::MatrixXd getStateLimits() const;
  Eigen::MatrixXd getActionLimits() const;

  /// Return the action which needs to be taken in the given state
  virtual Eigen::VectorXd getAction(const Eigen::VectorXd & state) = 0;

  // TODO incorporate the 'terminal action' flag?
  /// Add the sample to the sample collection and take any required action
  virtual void feed(const csa_mdp::Sample & sample);

  /// Update the content of the learner, this method should not be called during
  /// a trial because it is likely to require a lot of time
  virtual void internalUpdate() = 0;

  //TODO might be removed if RandomPolicy can be saved
  /// Has a policy already been computed by the learner ?
  virtual bool hasAvailablePolicy() = 0;

  /// Save current policy with the given prefix
  virtual void savePolicy(const std::string & prefix) = 0;

  /// Save current status with the given prefix
  virtual void saveStatus(const std::string & prefix) = 0;

protected:
  /// Acquired samples until now
  std::vector<Sample> samples;

private:
  /// Limits of state space used for learning of:
  /// - model dynamics
  /// - value function
  /// - policy
  Eigen::MatrixXd state_limits;
  /// Limits of the action space
  Eigen::MatrixXd action_limits;
};

}
