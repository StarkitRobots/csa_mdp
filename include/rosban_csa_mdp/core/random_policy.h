#pragma once

#include "rosban_csa_mdp/core/policy.h"

#include <random>

namespace csa_mdp
{

/// This class implements a policy as a set of regression forests, one for each dimension
class RandomPolicy : Policy
{
public:
  RandomPolicy();

  Eigen::VectorXd getRawAction(const Eigen::VectorXd &state) override;

  void to_xml(std::ostream & out) const override;
  void from_xml(TiXmlNode * node) override;
  std::string class_name() const override;

private:
  std::default_random_engine random_engine;
};

}
