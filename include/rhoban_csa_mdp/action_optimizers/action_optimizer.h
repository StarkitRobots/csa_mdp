#pragma once

#include "rhoban_csa_mdp/core/policy.h"
#include "rhoban_csa_mdp/core/problem.h"

#include "rhoban_utils/serialization/json_serializable.h"

#include <Eigen/Core>

#include <memory>

namespace csa_mdp
{

/// Describe the interface of an action optimizer
class ActionOptimizer : public rhoban_utils::JsonSerializable
{
public:
  ActionOptimizer();
  virtual ~ActionOptimizer();

  /// Set the maximal number of threads allowed
  void setNbThreads(int nb_threads);

  /// Try to find the best action with the given parameters
  /// if engine is note provided, it should handle its own
  virtual Eigen::VectorXd optimize(const Eigen::VectorXd & input,
                                   const Eigen::MatrixXd & action_limits,
                                   std::shared_ptr<const Policy> current_policy,
                                   Problem::ResultFunction result_function,
                                   Problem::ValueFunction value_function,
                                   double discount,
                                   std::default_random_engine * engine = nullptr) const = 0;

protected:
  int nb_threads;
};

}
