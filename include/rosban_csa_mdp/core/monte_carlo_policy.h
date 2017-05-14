#pragma once

#include "rosban_csa_mdp/core/policy.h"

namespace csa_mdp
{

class Problem;

/// In the MonteCarloPolicy, an optimization is performed on the first step
/// to be taken, then several other steps are performed using the 
/// default_policy (provided by the user). This procedure allows to perform
/// a local search while still benefiting of offline procedures for the
/// long-term behaviors.
///
/// TODO: accept a multiple step optimization
class MonteCarloPolicy : public Policy
{
public:
  virtual void init() override;

private:
  /// Engine used when none is provided
  std::default_random_engine internal_engine;

  /// Definition of the problem is required to simulate the behavior
  std::unique_ptr<Problem> problem;

  /// The policy which will be used once the optimization step has been performed
  std::unique_ptr<Policy> default_policy;

  /// Number of rollouts used to average the reward function
  int nb_rollouts;

  /// How many steps are taken in total
  int simulation_depth;

  /// The optimizer used for local search
  std::unique_ptr<rosban_bbo::Optimizer> optimizer;
}

}
