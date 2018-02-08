#include "rhoban_csa_mdp/solvers/mre.h"

#include "rhoban_fa/forest_approximator.h"
#include "rhoban_fa/function_approximator.h"

#include "rhoban_regression_forests/approximations/pwc_approximation.h"
#include "rhoban_random/tools.h"
#include "rhoban_regression_forests/tools/statistics.h"

#include "rhoban_utils/timing/benchmark.h"

#include <set>
#include <iostream>

using rhoban_utils::Benchmark;
using rhoban_utils::TimeStamp;

using rhoban_fa::ForestApproximator;
using rhoban_fa::FunctionApproximator;

using regression_forests::Forest;
using regression_forests::TrainingSet;
using namespace regression_forests::Statistics;

namespace csa_mdp
{

MRE::MRE()
  : plan_period(-1)
{
  // Init random engine
  random_engine = rhoban_random::getRandomEngine();
}

void MRE::setNbThreads(int new_nb_threads)
{
  Learner::setNbThreads(new_nb_threads);
  mrefpf_conf.nb_threads = new_nb_threads;
}

void MRE::feed(const Sample &s)
{
  if (!knownness_forest) {
    throw std::logic_error("MRE::feed: knownness_forest has not been initialized");
  }
  if (getActionLimits().size() !=1) {
    throw std::runtime_error("MRE::feed: not implemented for multiple actions problems");
  }

  int s_dim = getStateLimits().rows();
  int a_dim = getActionLimits()[0].rows();
  // Add the new 4 tuple
  samples.push_back(s);
  // Adding last_point to knownness tree
  Eigen::VectorXd knownness_point(s_dim + a_dim);
  knownness_point.segment(    0, s_dim) = s.state;
  knownness_point.segment(s_dim, a_dim) = s.action;
  knownness_forest->push(knownness_point);
  // Update policy if required
  if (plan_period > 0 && samples.size() % plan_period == 0)
  {
    internalUpdate();
  }
}

Eigen::VectorXd MRE::getAction(const Eigen::VectorXd &state)
{
  if (getActionLimits().size() !=1) {
    throw std::runtime_error("MRE::getAction: not implemented for multiple actions problems");
  }
  const Eigen::MatrixXd & limits = getActionLimits()[0];
  if (hasAvailablePolicy()) {

    Eigen::VectorXd action(policies.size());
    for (size_t i = 0; i < policies.size(); i++)
    {
      action(i) = policies[i]->getRandomizedValue(state, random_engine);
      double min = limits(i,0);
      double max = limits(i,1);
      // Ensuring that action is in the given bounds
      if (action(i) < min) action(i) = min;
      if (action(i) > max) action(i) = max;
    }
    return action;
  }
  return rhoban_random::getUniformSamples(limits, 1, &random_engine)[0];
}

void MRE::internalUpdate()
{
  if (getActionLimits().size() !=1) {
    throw std::runtime_error("MRE::getAction: not implemented for multiple actions problems");
  }
  const Eigen::MatrixXd & limits = getActionLimits()[0];

  // Updating the policy
  Benchmark::open("solver.solve");
  solver.solve(samples, terminal_function, mrefpf_conf);
  Benchmark::close();//true, -1);
  policies.clear();
  for (int dim = 0; dim < limits.rows(); dim++)
  {
    //TODO software design should really be improved
    policies.push_back(solver.stealPolicyForest(dim));
  }
  // Set time repartition
  time_repartition["QTS"] = mrefpf_conf.q_training_set_time;
  time_repartition["QET"] = mrefpf_conf.q_extra_trees_time;
  time_repartition["PTS"] = mrefpf_conf.p_training_set_time;
  time_repartition["PET"] = mrefpf_conf.p_extra_trees_time;
}

bool MRE::hasAvailablePolicy()
{
  return policies.size() > 0;
}

const regression_forests::Forest & MRE::getPolicy(int dim)
{
  return *(policies[dim]);
}

void MRE::savePolicy(const std::string &prefix)
{
  if (getActionLimits().size() !=1) {
    throw std::runtime_error("MRE::getAction: not implemented for multiple actions problems");
  }
  const Eigen::MatrixXd & limits = getActionLimits()[0];

  std::unique_ptr<ForestApproximator::Forests> forests(new ForestApproximator::Forests);
  for (int dim = 0; dim < limits.rows(); dim++)
  {
    forests->push_back(std::unique_ptr<Forest>(policies[dim]->clone()));
  }
#ifdef RHOBAN_RF_USES_GP
  if (mrefpf_conf.gp_policies) {
    throw std::logic_error("Saving gp_policies with MRE is not implemented yet");
  }
#endif
  ForestApproximator fa(std::move(forests), 0);
  fa.save(prefix + "policy.data");
}

void MRE::saveValue(const std::string &prefix)
{
  solver.getValueForest().save(prefix + "q_value.data");
}

void MRE::saveKnownnessTree(const std::string &prefix)
{
  knownness_forest->checkConsistency();
  std::unique_ptr<regression_forests::Forest> forest;
  forest = knownness_forest->convertToRegressionForest();
  forest->save(prefix + "knownness.data");
}

void MRE::saveStatus(const std::string &prefix)
{
  savePolicy(prefix);
  saveValue(prefix);
  saveKnownnessTree(prefix);
}

void MRE::setStateLimits(const Eigen::MatrixXd & limits)
{
  Learner::setStateLimits(limits);
  mrefpf_conf.setStateLimits(limits);
  updateQSpaceLimits();
}

void MRE::setActionLimits(const std::vector<Eigen::MatrixXd> & limits)
{
  if (limits.size() != 1) {
    throw std::runtime_error("MRE::setActionLimits: not implemented for multiple actions problems");
  }

  Learner::setActionLimits(limits);
  mrefpf_conf.setActionLimits(limits[0]);
  updateQSpaceLimits();
}

void MRE::updateQSpaceLimits()
{
  Eigen::MatrixXd q_space = mrefpf_conf.getInputLimits();
  knownness_forest = std::shared_ptr<KnownnessForest>(new KnownnessForest(q_space,
                                                                          knownness_conf));
  solver.setKnownnessFunc(knownness_forest);
}

std::string MRE::getClassName() const
{
  return "MRE";
}

Json::Value MRE::toJson() const
{
  Json::Value v;
  v["plan_period"] = plan_period;
  v["mrefpf_conf"] = mrefpf_conf.toJson();
  v["knownness_conf"] = knownness_conf.toJson();
  return v;
}

void MRE::fromJson(const Json::Value & v, const std::string & dir_name)
{
  (void)dir_name;
  rhoban_utils::tryRead(v, "plan_period", &plan_period);
  mrefpf_conf.read(v, "mrefpf_conf");
  knownness_conf.tryRead(v, "knownness_conf");
}

}
