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

  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

protected:
  int nb_threads;
};

}
