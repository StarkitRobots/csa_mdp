#pragma once

#include "rosban_csa_mdp/core/sample.h"

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

#include <functional>
#include <random>

namespace csa_mdp
{

class Problem : public rosban_utils::Serializable
{
public:
  /// Return true if state is terminal
  typedef std::function<bool(const Eigen::VectorXd &state)> TerminalFunction;
  /// Return a value associate to the state
  typedef std::function<double(const Eigen::VectorXd &state)> ValueFunction;
  /// Return action for given state
  typedef std::function<Eigen::VectorXd(const Eigen::VectorXd &state)> Policy;
  /// Sample successor state from a couple (state, action) using provided random engine
  typedef std::function<Eigen::VectorXd(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &action,
                                        std::default_random_engine * engine)> TransitionFunction;
  /// Return Reward for the given triplet (state, action, next_state)
  typedef std::function<double(const Eigen::VectorXd &state,
                               const Eigen::VectorXd &action,
                               const Eigen::VectorXd &next_state)> RewardFunction;

private:
  Eigen::MatrixXd state_limits;
  Eigen::MatrixXd action_limits;

  std::vector<std::string> state_names;
  std::vector<std::string> action_names;

  std::vector<std::uniform_real_distribution<double>> state_distribution;
  std::vector<std::uniform_real_distribution<double>> action_distribution;

protected:
  std::default_random_engine random_engine;

public:
  Problem();
  virtual ~Problem();

  RewardFunction getRewardFunction() const;
  TransitionFunction getTransitionFunction() const;
  TerminalFunction getTerminalFunction() const;

  int stateDims() const;
  int actionDims() const;

  const Eigen::MatrixXd & getStateLimits() const;
  const Eigen::MatrixXd & getActionLimits() const;

  /// Also reset state names
  void setStateLimits(const Eigen::MatrixXd &new_limits);

  /// Also reset action names
  void setActionLimits(const Eigen::MatrixXd &new_limits);


  /// Set the names of the states to "state_0, state_1, ..."
  void resetStateNames();

  /// Set the names of the states to "action_0, action_1, ..."
  void resetActionNames();

  /// To call after setting properly the limits
  /// throw a runtime_error if names size is not appropriate
  void setStateNames(const std::vector<std::string> &names);

  /// To call after setting properly the limits
  /// throw a runtime_error if names size is not appropriate
  void setActionNames(const std::vector<std::string> &names);

  const std::vector<std::string> & getStateNames() const;
  const std::vector<std::string> & getActionNames() const;

  /// Which state dimensions are used as input for learning (default is all)
  virtual std::vector<int> getLearningDimensions() const;
  
  virtual bool isTerminal(const Eigen::VectorXd & state) const = 0;

  /// This function is not allowed to be stochastic
  virtual double getReward(const Eigen::VectorXd &state,
                           const Eigen::VectorXd &action,
                           const Eigen::VectorXd &dst) const = 0;

  /// This function  uses the inner random engine
  Eigen::VectorXd getSuccessor(const Eigen::VectorXd &state,
                               const Eigen::VectorXd &action);

  /// Use an external random engine
  virtual Eigen::VectorXd getSuccessor(const Eigen::VectorXd &state,
                                       const Eigen::VectorXd &action,
                                       std::default_random_engine * engine)  const = 0;

  /// Provide a random state in state_limits (uniformous distribution)
  Eigen::VectorXd getRandomState();
  /// Provide a random action in action_limits (uniformous distribution)
  Eigen::VectorXd getRandomAction();
  /// Provide a sample starting from state and using a random action
  Sample getRandomSample(const Eigen::VectorXd &state);
  /// Compute a random trajectory starting at the given state
  std::vector<Sample> getRandomTrajectory(const Eigen::VectorXd &initial_state,
                                          int max_length);
  /// Compute several random trajectories starting from the same state
  std::vector<Sample> getRandomBatch(const Eigen::VectorXd &initial_state,
                                     int max_length,
                                     int nb_trajectories);

  /// Provide a sample starting from state and using a specified action
  Sample getSample(const Eigen::VectorXd &state,
                   const Eigen::VectorXd &action);
  /// Simulate a trajectory using a given policy
  std::vector<Sample> simulateTrajectory(const Eigen::VectorXd &initial_state,
                                         int max_length,
                                         Policy p);
};

}
