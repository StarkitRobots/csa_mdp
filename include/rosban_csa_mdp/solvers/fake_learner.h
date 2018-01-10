#pragma once

#include "rosban_csa_mdp/solvers/learner.h"

#include "rosban_csa_mdp/core/policy.h"

#include <memory>

namespace csa_mdp
{

class FakeLearner : public Learner
{
public:
  virtual Eigen::VectorXd getAction(const Eigen::VectorXd & state);
  virtual void internalUpdate();
  virtual bool hasAvailablePolicy();
  virtual void savePolicy(const std::string & prefix);
  virtual void saveStatus(const std::string & prefix);

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

  /// Reset effects on policy if necessary
  virtual void endRun() override;

  /// Update number of threads to be used for the policy too
  virtual void setNbThreads(int nb_threads) override;

protected:
  std::unique_ptr<Policy> policy;
};

}
