#pragma once

#include "rosban_csa_mdp/core/policy.h"

#include <random>

namespace csa_mdp
{

class RandomPolicy : public Policy
{
public:
  RandomPolicy();

  Eigen::VectorXd getRawAction(const Eigen::VectorXd &state) override;

  Eigen::VectorXd getRawAction(const Eigen::VectorXd &state,
                               std::default_random_engine * engine) const override;

  std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

private:
  std::default_random_engine random_engine;
};

}
