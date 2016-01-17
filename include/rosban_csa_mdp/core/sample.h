#pragma once

#include <Eigen/Core>

namespace csa_mdp
{

/// This class wraps a 4-tuple (s,a, s', r) which corresponds to a sample of a continuous state mdp
class Sample
{
public:
  Eigen::VectorXd state;
  Eigen::VectorXd action;
  Eigen::VectorXd next_state;
  double reward;
  
  Sample();
  Sample(const Eigen::VectorXd &state,
         const Eigen::VectorXd &action,
         const Eigen::VectorXd &next_state,
         double reward);
};

}
