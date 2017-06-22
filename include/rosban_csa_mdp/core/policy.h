#pragma once

#include "rosban_fa/fa_tree.h"

#include "rosban_utils/serializable.h"
#include "rosban_utils/stream_serializable.h"

#include <Eigen/Core>

#include <memory>
#include <random>

namespace csa_mdp
{

class Policy : public rosban_utils::Serializable
{
public:
  Policy();

  /// Some policies might have special behaviors at the beginning of a trial
  virtual void init();

  /// Define the minimal and maximal limits for the policy along each dimensions
  virtual void setActionLimits(const std::vector<Eigen::MatrixXd> & limits);

  Eigen::VectorXd boundAction(const Eigen::VectorXd &raw_action) const;
  
  /// Retrieve the action corresponding to the given state
  Eigen::VectorXd getAction(const Eigen::VectorXd &state);

  /// Retrieve the action corresponding to the given state
  Eigen::VectorXd getAction(const Eigen::VectorXd &state,
                            std::default_random_engine * engine) const;

  /// Retrieve the raw action corresponding to the given state
  virtual Eigen::VectorXd getRawAction(const Eigen::VectorXd &state);

  /// Retrieve the raw action corresponding to the given state
  virtual Eigen::VectorXd getRawAction(const Eigen::VectorXd &state,
                                       std::default_random_engine * engine) const = 0;

  /// Return an approximation of the current policy as a FATree
  virtual std::unique_ptr<rosban_fa::FATree> extractFATree() const;

protected:
  std::vector<Eigen::MatrixXd> action_limits;

  /// Required when getRawAction doest not provide a random engine
  std::default_random_engine internal_random_engine;
};

}
