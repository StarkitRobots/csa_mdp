#include "rosban_csa_mdp/solvers/mre.h"

#include "rosban_regression_forests/approximations/pwc_approximation.h"
#include "rosban_regression_forests/tools/random.h"
#include "rosban_regression_forests/tools/statistics.h"

#include <set>
#include <iostream>

using regression_forests::TrainingSet;
using namespace regression_forests::Statistics;

namespace csa_mdp
{

MRE::CustomFPF::CustomFPF(const Eigen::MatrixXd &q_space,
                          int max_points,
                          double reward_max,
                          int nb_trees,
                          KnownnessTree::Type type)
  : r_max(reward_max)
{
  KnownnessForest::Config forest_config;
  forest_config.nb_trees = nb_trees;
  forest_config.tree_conf.max_points = max_points;
  forest_config.tree_conf.type = type;
  knownness_forest = KnownnessForest(q_space, forest_config);
}

void MRE::CustomFPF::push(const Eigen::VectorXd &q_point)
{
  knownness_forest.push(q_point);
}

double MRE::CustomFPF::getKnownness(const Eigen::VectorXd& point) const
{
  return knownness_forest.getValue(point);
}

std::unique_ptr<regression_forests::Forest> MRE::CustomFPF::getKnownnessForest()
{
  return knownness_forest.convertToRegressionForest();
}

TrainingSet MRE::CustomFPF::getTrainingSet(const std::vector<Sample>& samples,
                                           std::function<bool(const Eigen::VectorXd&)> is_terminal)
{
  // Removing samples which have the same starting state
  std::vector<Sample> filtered_samples;
  for (const Sample & new_sample : samples)
  {
    bool found_similar = false;
    double tolerance = std::pow(10,-6);
    for (const Sample & known_sample : filtered_samples)
    {
      Eigen::VectorXd state_diff = known_sample.state - new_sample.state;
      Eigen::VectorXd action_diff = known_sample.action - new_sample.action;
      // Which is the highest difference between the two samples?
      double max_diff = std::max(state_diff.lpNorm<Eigen::Infinity>(),
                                 action_diff.lpNorm<Eigen::Infinity>());
      if (max_diff < tolerance)
      {
        found_similar = true;
        break;
      }
    }
    // Do not add similar samples
    if (found_similar) continue;
    filtered_samples.push_back(new_sample);
  }
  TrainingSet original_ts = FPF::getTrainingSet(filtered_samples, is_terminal);
  TrainingSet new_ts(original_ts.getInputDim());
  for (size_t i = 0; i < original_ts.size(); i++)
  {
    // Extracting information from original sample
    const regression_forests::Sample & original_sample = original_ts(i);
    Eigen::VectorXd input = original_sample.getInput();
    double reward         = original_sample.getOutput();
    // Getting knownness of the input
    double knownness = getKnownness(input);
    double new_reward = reward * knownness + r_max * (1 - knownness);
    new_ts.push(regression_forests::Sample(input, new_reward));
  }
  return new_ts;
}

MRE::Config::Config()
{
}

std::string MRE::Config::class_name() const
{
  return "Config";
}

void MRE::Config::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<double>("reward_max" , reward_max , out);
  rosban_utils::xml_tools::write<int>   ("max_points" , max_points , out);
  rosban_utils::xml_tools::write<int>   ("plan_period", plan_period, out);
  rosban_utils::xml_tools::write<int>   ("nb_trees"   , nb_trees   , out);
  std::string tree_type_str = to_string(tree_type);
  rosban_utils::xml_tools::write<std::string>("tree_type", tree_type_str, out);
  fpf_conf.write("fpf_conf", out);
}

void MRE::Config::from_xml(TiXmlNode *node)
{
  reward_max  = rosban_utils::xml_tools::read<double>(node, "reward_max" );
  max_points  = rosban_utils::xml_tools::read<int>   (node, "max_points" );
  plan_period = rosban_utils::xml_tools::read<int>   (node, "plan_period");
  nb_trees    = rosban_utils::xml_tools::read<int>   (node, "nb_trees"   );
  std::string tree_type_str;
  tree_type_str = rosban_utils::xml_tools::read<std::string>(node, "tree_type");
  tree_type = loadType(tree_type_str);
  fpf_conf.read(node, "fpf_conf");
}

MRE::MRE(const MRE::Config &config,
         std::function<bool(const Eigen::VectorXd &)> is_terminal_)
  : MRE(config.fpf_conf.getStateLimits(),
        config.fpf_conf.getActionLimits(),
        config.max_points,
        config.reward_max,
        config.plan_period,
        config.nb_trees,
        config.tree_type,
        config.fpf_conf,
        is_terminal_)
{
}

MRE::MRE(const Eigen::MatrixXd &state_space_,
         const Eigen::MatrixXd &action_space_,
         int max_points,
         double reward_max,
         int plan_period_,
         int nb_trees,
         KnownnessTree::Type knownness_tree_type,
         const FPF::Config &fpf_conf,
         std::function<bool(const Eigen::VectorXd &)> is_terminal_)
  : plan_period(plan_period_),
    is_terminal(is_terminal_),
    solver(Eigen::MatrixXd(0,0),0,0, 0, knownness_tree_type),
    state_space(state_space_),
    action_space(action_space_)
{
  int s_dim = state_space.rows();
  int a_dim = action_space.rows();
  Eigen::MatrixXd q_space(s_dim + a_dim, 2);
  q_space.block(    0, 0, s_dim, 2) = state_space;
  q_space.block(s_dim, 0, a_dim, 2) = action_space;
  solver = CustomFPF(q_space, max_points, reward_max, nb_trees, knownness_tree_type);
  solver.conf = fpf_conf;
  random_engine = regression_forests::get_random_engine();
}

void MRE::feed(const Sample &s)
{
  int s_dim = state_space.rows();
  int a_dim = action_space.rows();
  // Add the new 4 tuple
  samples.push_back(s);
  // Adding last_point to knownness tree
  Eigen::VectorXd knownness_point(s_dim + a_dim);
  knownness_point.segment(    0, s_dim) = s.state;
  knownness_point.segment(s_dim, a_dim) = s.action;
  solver.push(knownness_point);
  // Update policy if required
  if (plan_period > 0 && samples.size() % plan_period == 0)
  {
    updatePolicy();
  }
}

Eigen::VectorXd MRE::getAction(const Eigen::VectorXd &state)
{
  if (policies.size() > 0) {
    Eigen::VectorXd action(policies.size());
    for (size_t i = 0; i < policies.size(); i++)
    {
      action(i) = policies[i]->getValue(state);
      double min = action_space(i,0);
      double max = action_space(i,1);
      if (action(i) < min) action(i) = min;
      if (action(i) > max) action(i) = max;
    }
    return action;
  }
  return regression_forests::getUniformSamples(action_space, 1, &random_engine)[0];
}

void MRE::updatePolicy()
{
  solver.solve(samples, is_terminal);
  policies.clear();
  for (int dim = 0; dim < action_space.rows(); dim++)
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
  for (int dim = 0; dim < action_space.rows(); dim++)
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
  forest = solver.getKnownnessForest();
  forest->save(prefix + "knownness.data");
}

void MRE::saveStatus(const std::string &prefix)
{
  savePolicies(prefix);
  saveValue(prefix);
  saveKnownnessTree(prefix);
}


double MRE::getQValueTime() const
{
  return solver.conf.q_value_time;
}

double MRE::getPolicyTime() const
{
  return solver.conf.policy_time;
}

}
