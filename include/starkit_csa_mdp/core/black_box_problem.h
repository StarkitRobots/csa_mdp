#pragma once
#include "starkit_csa_mdp/core/problem.h"

namespace csa_mdp
{
class BlackBoxProblem : public csa_mdp::Problem
{
public:
  /// By which state should the episode start
  virtual Eigen::VectorXd getStartingState(std::default_random_engine* engine) const = 0;
};

}  // namespace csa_mdp
