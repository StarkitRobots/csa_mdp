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
  typedef std::function<double(const Eigen::VectorXd & state)> ValueFunction;

  /// Predict the reward using 'local planning' fo the next 'nb steps', after
  /// those 'nb steps', the value function is used to approximate the long term
  /// reward. Some methods might use the current policy to improve their long
  /// term predictions
  virtual void predict(const Eigen::VectorXd & input,
                       std::shared_ptr<const Policy> policy,
                       int nb_steps,
                       std::shared_ptr<Problem> model,//TODO: Model class ?
                       RewardFunction reward_function,
                       ValueFunction value_function,
                       double discount,
                       double * mean,
                       double * var) = 0;
};

}
