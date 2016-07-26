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
  /// Predict the reward using 'local planning' fo the next 'nb steps', after
  /// those 'nb steps', the value function is used to approximate the long term
  /// reward. Some methods might use the current policy to improve their long
  /// term predictions
  virtual void predict(const Eigen::VectorXd & input,
                       std::shared_ptr<const Policy> policy,
                       int nb_steps,
                       std::shared_ptr<Problem> model,//TODO: Model class ?
                       Problem::RewardFunction reward_function,
                       Problem::ValueFunction value_function,
                       Problem::TerminalFunction terminal_function,
                       double discount,
                       double * mean,
                       double * var) = 0;
};

}
