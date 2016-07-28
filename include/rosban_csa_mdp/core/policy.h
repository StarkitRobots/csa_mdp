#pragma once

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

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
  virtual void setActionLimits(const Eigen::MatrixXd &limits);

  Eigen::VectorXd boundAction(const Eigen::VectorXd &raw_action) const;
  
  /// Retrieve the action corresponding to the given state
  Eigen::VectorXd getAction(const Eigen::VectorXd &state);

  /// Retrieve the action corresponding to the given state
  Eigen::VectorXd getAction(const Eigen::VectorXd &state,
                            std::default_random_engine * engine) const;

  /// Retrieve the raw action correspoding to the given state
  virtual Eigen::VectorXd getRawAction(const Eigen::VectorXd &state) = 0;

  /// Retrieve the raw action correspoding to the given state
  virtual Eigen::VectorXd getRawAction(const Eigen::VectorXd &state,
                                       std::default_random_engine * engine) const = 0;

protected:
  Eigen::MatrixXd action_limits;

};

}
