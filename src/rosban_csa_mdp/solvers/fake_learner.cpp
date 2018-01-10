#include "rosban_csa_mdp/solvers/fake_learner.h"

#include "rosban_csa_mdp/core/policy_factory.h"

namespace csa_mdp
{

Eigen::VectorXd FakeLearner::getAction(const Eigen::VectorXd & state)
{
  policy->setActionLimits(getActionLimits());
  return policy->getAction(state);
}

void FakeLearner::internalUpdate() {}

bool FakeLearner::hasAvailablePolicy() { return false; }
void FakeLearner::savePolicy(const std::string & prefix) {(void)prefix;}
void FakeLearner::saveStatus(const std::string & prefix) {(void)prefix;}

std::string FakeLearner::getClassName() const
{
  return "FakeLearner";
}

void FakeLearner::to_xml(std::ostream &out) const
{
  policy->factoryWrite("policy", out);
}

void FakeLearner::from_xml(TiXmlNode *node)
{
  policy = PolicyFactory().read(node, "policy");
}

void FakeLearner::endRun()
{
  if (policy) policy->init();
}

void FakeLearner::setNbThreads(int nb_threads)
{
  Learner::setNbThreads(nb_threads);
  policy->setNbThreads(nb_threads);
}

}
