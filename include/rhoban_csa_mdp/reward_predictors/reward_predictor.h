#pragma once

#include "rhoban_csa_mdp/core/policy.h"
#include "rhoban_csa_mdp/core/problem.h"

#include "rhoban_utils/serialization/json_serializable.h"

#include <Eigen/Core>

#include <memory>

namespace csa_mdp
{

class RewardPredictor : public rhoban_utils::JsonSerializable
{
public:

  RewardPredictor();
  virtual ~RewardPredictor();

  /// Predict the expected value at infinite horizon from the given state:
  /// - The value function might be used to approximate after a given number of steps
  /// - Some methods might use the current policy to improve their long term predictions
  virtual void predict(const Eigen::VectorXd & input,
                       const Policy & policy,
                       Problem::ResultFunction result_function,
                       Problem::ValueFunction value_function,
                       double discount,
                       double * mean,
                       double * var,
                       std::default_random_engine * engine) = 0;

  virtual void setNbThreads(int nb_threads);

  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

protected:
  int nb_threads;
};

}
