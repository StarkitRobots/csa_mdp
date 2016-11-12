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

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

  /// Reset effects on policy if necessary
  virtual void endRun() override;

protected:
  std::unique_ptr<Policy> policy;
};

}
