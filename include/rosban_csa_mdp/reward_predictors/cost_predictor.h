#pragma once

#include "rosban_utils/serializable.h"

namespace csa_mdp
{

class RewardPredictor : public rosban_utils::Serializable
{
public:

  typedef std::function<double(const Eigen::VectorXd &, const Eigen::VectorXd &)> RewardFunction;

  ///TODO:
  /// - Think about transfer of ownership
  /// - Think about specific classes (Policy, Model)
  /// - Const properties and random
  /// - Parallelization <- const are required for parallelization
  virtual void predict(const Eigen::VectorXd & input,
                       std::shared_ptr<const Policy> policy,
                       int nb_steps,
                       std::shared_ptr<const Problem> model,//TODO: Model class ?
                       RewardFunction reward_function,
                       double discount,
                       double * mean,
                       double * var) = 0;
};

}
