#pragma once

#include "starkit_csa_mdp/core/policy.h"
#include "starkit_csa_mdp/core/problem.h"

#include <random>

namespace csa_mdp
{
class OpportunistPolicy : public csa_mdp::Policy
{
public:
  OpportunistPolicy();

  void setActionLimits(const std::vector<Eigen::MatrixXd>& limits) override;

  /// Get best action among policies according to rollout and given problem
  Eigen::VectorXd getRawAction(const Eigen::VectorXd& state,
                               std::default_random_engine* external_engine) const override;

  std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

private:
  /// Given problem
  std::unique_ptr<Problem> problem;

  /// Available policies
  std::vector<std::unique_ptr<Policy>> policies;

  /// Number of rollouts performed by
  int nb_rollouts;

  /// Horizon used for the rollout
  int horizon;
};

}  // namespace csa_mdp
