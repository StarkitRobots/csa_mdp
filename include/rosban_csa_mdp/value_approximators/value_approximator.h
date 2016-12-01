#pragma once

#include "rosban_csa_mdp/core/policy.h"
#include "rosban_csa_mdp/core/problem.h"

#include "rosban_fa/function_approximator.h"

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

#include <memory>

namespace csa_mdp
{

class ValueApproximator : public rosban_utils::Serializable {
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

  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

protected:
  /// Number of threads allowed to run simultaneously
  int nb_threads;
};

}
