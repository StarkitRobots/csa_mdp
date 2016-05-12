#include "rosban_csa_mdp/solvers/mre.h"

#include "rosban_regression_forests/approximations/pwc_approximation.h"
#include "rosban_random/tools.h"
#include "rosban_regression_forests/tools/statistics.h"

#include "rosban_utils/benchmark.h"

#include <set>
#include <iostream>

using rosban_utils::Benchmark;
using rosban_utils::TimeStamp;

using regression_forests::TrainingSet;
using namespace regression_forests::Statistics;

namespace csa_mdp
{

MRE::Config::Config()
  : plan_period(-1)
{
}

std::string MRE::Config::class_name() const
{
  return "MREConfig";
}

void MRE::Config::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("plan_period", plan_period, out);
  mrefpf_conf.write("mrefpf_conf", out);
  knownness_conf.write("knownness_conf", out);
}

void MRE::Config::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>(node, "plan_period", plan_period);
  mrefpf_conf.read(node, "mrefpf_conf");
  knownness_conf.tryRead(node, "knownness_conf");
}

MRE::MRE(const MRE::Config &conf_,
         std::function<bool(const Eigen::VectorXd &)> is_terminal_)
  : conf(conf_),
    is_terminal(is_terminal_)
{
  // Init Knownness Forest
  Eigen::MatrixXd q_space = conf.mrefpf_conf.getInputLimits();
  knownness_forest = std::shared_ptr<KnownnessForest>(new KnownnessForest(q_space, conf.knownness_conf));
  // Init random engine
  random_engine = rosban_random::getRandomEngine();
}

void MRE::feed(const Sample &s)
{
  int s_dim = getStateSpace().rows();
  int a_dim = getActionSpace().rows();
  // Add the new 4 tuple
  samples.push_back(s);
  // Adding last_point to knownness tree
  Eigen::VectorXd knownness_point(s_dim + a_dim);
  knownness_point.segment(    0, s_dim) = s.state;
  knownness_point.segment(s_dim, a_dim) = s.action;
  knownness_forest->push(knownness_point);
  // Update policy if required
  if (conf.plan_period > 0 && samples.size() % conf.plan_period == 0)
  {
    updatePolicy();
  }
  // Initializing solver
  solver = MREFPF(knownness_forest);
}

Eigen::VectorXd MRE::getAction(const Eigen::VectorXd &state)
{
  if (policies.size() > 0) {
    Eigen::VectorXd action(policies.size());
    for (size_t i = 0; i < policies.size(); i++)
    {
      action(i) = policies[i]->getRandomizedValue(state, random_engine);
      double min = getActionSpace()(i,0);
      double max = getActionSpace()(i,1);
      // Ensuring that action is in the given bounds
      if (action(i) < min) action(i) = min;
      if (action(i) > max) action(i) = max;
    }
    return action;
  }
  return rosban_random::getUniformSamples(getActionSpace(), 1, &random_engine)[0];
}

void MRE::updatePolicy()
{
  // Updating the policy
  Benchmark::open("solver.solve");
  solver.solve(samples, is_terminal, conf.mrefpf_conf);
  Benchmark::close();//true, -1);
  policies.clear();
  for (int dim = 0; dim < getActionSpace().rows(); dim++)
  {
    //TODO software design should really be improved
    policies.push_back(solver.stealPolicyForest(dim));
  }
}

const regression_forests::Forest & MRE::getPolicy(int dim)
{
  return *(policies[dim]);
}

void MRE::savePolicies(const std::string &prefix)
{
  for (int dim = 0; dim < getActionSpace().rows(); dim++)
  {
    policies[dim]->save(prefix + "policy_d" + std::to_string(dim) + ".data");
  }
}

void MRE::saveValue(const std::string &prefix)
{
  solver.getValueForest().save(prefix + "q_value.data");
}

void MRE::saveKnownnessTree(const std::string &prefix)
{
  std::unique_ptr<regression_forests::Forest> forest;
  forest = knownness_forest->convertToRegressionForest();
  forest->save(prefix + "knownness.data");
}

void MRE::saveStatus(const std::string &prefix)
{
  savePolicies(prefix);
  saveValue(prefix);
  saveKnownnessTree(prefix);
}

double MRE::getQValueTrainingSetTime() const
{
  return conf.mrefpf_conf.q_training_set_time;
}

double MRE::getQValueExtraTreesTime() const
{
  return conf.mrefpf_conf.q_extra_trees_time;
}

double MRE::getPolicyTrainingSetTime() const
{
  return conf.mrefpf_conf.p_training_set_time;
}

double MRE::getPolicyExtraTreesTime() const
{
  return conf.mrefpf_conf.p_extra_trees_time;
}

const Eigen::MatrixXd & MRE::getStateSpace()
{
  return conf.mrefpf_conf.getStateLimits();
}

const Eigen::MatrixXd & MRE::getActionSpace()
{
  return conf.mrefpf_conf.getActionLimits();
}

}
