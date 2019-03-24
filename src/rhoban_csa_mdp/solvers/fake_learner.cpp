#include "rhoban_csa_mdp/solvers/fake_learner.h"

#include "rhoban_csa_mdp/core/policy_factory.h"

namespace csa_mdp
{
Eigen::VectorXd FakeLearner::getAction(const Eigen::VectorXd& state)
{
  policy->setActionLimits(getActionLimits());
  return policy->getAction(state);
}

void FakeLearner::internalUpdate()
{
}

bool FakeLearner::hasAvailablePolicy()
{
  return false;
}
void FakeLearner::savePolicy(const std::string& prefix)
{
  (void)prefix;
}
void FakeLearner::saveStatus(const std::string& prefix)
{
  (void)prefix;
}

std::string FakeLearner::getClassName() const
{
  return "FakeLearner";
}

Json::Value FakeLearner::toJson() const
{
  Json::Value v;
  v["policy"] = policy->toFactoryJson();
  return v;
}

void FakeLearner::fromJson(const Json::Value& v, const std::string& dir_name)
{
  policy = PolicyFactory().read(v, "policy", dir_name);
}

void FakeLearner::endRun()
{
  if (policy)
    policy->init();
}

void FakeLearner::setNbThreads(int nb_threads)
{
  Learner::setNbThreads(nb_threads);
  policy->setNbThreads(nb_threads);
}

}  // namespace csa_mdp
