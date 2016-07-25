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

  typedef std::function<double(const Eigen::VectorXd & state,
                               const Eigen::VectorXd & action,
                               const Eigen::VectorXd & next_state)> RewardFunction;

  ///TODO:
  /// - Think about transfer of ownership
  /// - Think about specific classes (Policy, Model)
  /// - Const properties and random
  /// - Parallelization <- const are required for parallelization
  virtual void predict(const Eigen::VectorXd & input,
                       std::shared_ptr<const Policy> policy,
                       int nb_steps,
                       std::shared_ptr<Problem> model,//TODO: Model class ?
                       RewardFunction reward_function,
                       double discount,
                       double * mean,
                       double * var) = 0;
};

}
