#pragma once

#include "rosban_csa_mdp/core/policy.h"
#include "rosban_csa_mdp/core/problem.h"

#include "rosban_fa/function_approximator.h"

#include "rhoban_utils/serialization/json_serializable.h"

#include <Eigen/Core>

#include <memory>

namespace csa_mdp
{

class ValueApproximator : public rhoban_utils::JsonSerializable {
public:

  ValueApproximator();
  virtual ~ValueApproximator();

  /// Train the value function for the given problem
  virtual std::unique_ptr<rosban_fa::FunctionApproximator>
  train(const Policy & policy,
        const Problem & problem,
        Problem::ValueFunction current_value_function,
        double discount,
        std::default_random_engine * engine) = 0;

  virtual void setNbThreads(int nb_threads);

  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

protected:
  /// Number of threads allowed to run simultaneously
  int nb_threads;
};

}
