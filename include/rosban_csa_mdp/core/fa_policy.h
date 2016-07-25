#pragma once

#include "rosban_csa_mdp/core/policy.h"

#include "rosban_fa/function_approximator.h"

#include <memory>
#include <random>

namespace csa_mdp
{

/// This class implements a policy using a function approximator
class FAPolicy : public Policy
{
public:
  FAPolicy(std::unique_ptr<rosban_fa::FunctionApproximator> fa);

  Eigen::VectorXd getRawAction(const Eigen::VectorXd &state) override;
  Eigen::VectorXd getRawAction(const Eigen::VectorXd &state,
                               std::default_random_engine * engine) const override;

  void to_xml(std::ostream & out) const override;
  void from_xml(TiXmlNode * node) override;
  std::string class_name() const override;

private:
  /// The policies
  std::unique_ptr<rosban_fa::FunctionApproximator> fa;

  /// Is the noise applied when requesting raw action?
  bool apply_noise;

  /// Random generator used to select noisy actions
  std::default_random_engine engine;
};

}
