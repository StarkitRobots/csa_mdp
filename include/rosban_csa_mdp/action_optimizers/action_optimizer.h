#pragma once

#include "rosban_csa_mdp/core/policy.h"
#include "rosban_csa_mdp/core/problem.h"

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

#include <memory>

namespace csa_mdp
{

/// Describe the interface of an action optimizer
class ActionOptimizer : public rosban_utils::Serializable
{
public:

  typedef std::function<double(const Eigen::VectorXd & state,
                               const Eigen::VectorXd & action,
                               const Eigen::VectorXd & next_state)> RewardFunction;
  typedef std::function<double(const Eigen::VectorXd & state)> ValueFunction;

  /// Try to find the best action with the given parameters
  virtual Eigen::VectorXd optimize(const Eigen::VectorXd & input,
                                   std::shared_ptr<const Policy> current_policy,
                                   std::shared_ptr<Problem> model,//TODO: Model class ?
                                   RewardFunction reward_function,
                                   ValueFunction value_function,
                                   double discount) = 0;
};

}