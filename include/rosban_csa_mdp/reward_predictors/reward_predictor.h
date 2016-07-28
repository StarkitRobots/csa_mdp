#pragma once

#include "rosban_csa_mdp/core/policy.h"
#include "rosban_csa_mdp/core/problem.h"

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

#include <memory>

namespace csa_mdp
{

class RewardPredictor : public rosban_utils::Serializable
{
public:
  /// Predict the expected value at infinite horizon from the given state:
  /// - The value function might be used to approximate after a given number of steps
  /// - Some methods might use the current policy to improve their long term predictions
  virtual void predict(const Eigen::VectorXd & input,
                       std::shared_ptr<const Policy> policy,
                       std::shared_ptr<Problem> model,//TODO: Model class ?
                       Problem::RewardFunction reward_function,
                       Problem::ValueFunction value_function,
                       Problem::TerminalFunction terminal_function,
                       double discount,
                       double * mean,
                       double * var) = 0;
};

}
