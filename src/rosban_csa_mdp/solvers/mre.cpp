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

MRE::KnownnessTree::KnownnessTree(const Eigen::MatrixXd& space, int maxPoints, Type type_)
  : tree(space), v(maxPoints), nextSplitDim(0), nbPoints(0), type(type_)
{
  random_engine = regression_forests::get_random_engine();
}

void MRE::KnownnessTree::push(const Eigen::VectorXd& point)
{
  // Checking if the point is in the tree space
  const Eigen::MatrixXd &tree_space = tree.getSpace();
  for (int dim = 0; dim < point.rows(); dim++)
  {
    if (point(dim) < tree_space(dim,0) || point(dim) > tree_space(dim,1))
    {
      std::ostringstream oss;
      oss << "Point is outside of space!" << std::endl
          << "P: " << point.transpose() << std::endl
          << "Space:" << std::endl << tree_space << std::endl;
      throw std::runtime_error(oss.str());
    }
  }
  // Pushing point
  kd_trees::KdNode * leafNode = tree.getLeaf(point);
  Eigen::MatrixXd leaf_space = tree.getSpace(point);
  // Special case on random, avoid reaching a high density (Disabled)
  //if (type == Type::Random)
  //{
  //    double local_size = 1.0;
  //    double global_size = 1.0;
  //    for (int dim = 0; dim < leaf_space.rows(); dim++)
  //    {
  //      local_size  *= leaf_space(dim,1) - leaf_space(dim,0);
  //      global_size *= tree_space(dim,1) - tree_space(dim,0);
  //    }
  //    int local_points = leafNode->getPoints().size();
  //    double local_density  = local_points / local_size;
  //    double global_density = nbPoints / global_size;
  //    double density_ratio  = local_density / global_density;
  //    if (density_ratio > std::pow(10,6))
  //    {
  //      throw std::runtime_error("Leaf has already a high density");
  //    }
  //}
  leafNode->push(point);
  int leafCount = leafNode->getPoints().size();
  if (leafCount > v) {
    int split_dim = -1;
    double split_val = 0;
    switch(type)
    {
      case Type::Original:
        split_dim = nextSplitDim;
        split_val = (leaf_space(split_dim, 0) + leaf_space(split_dim,1)) / 2;
        break;
      case Type::Test:
      {
        const Eigen::MatrixXd &space = tree.getSpace();
        double highest_ratio = 0;
        split_dim = -1;
        for (int dim = 0; dim < leaf_space.rows(); dim++)
        {
          double leaf_size = leaf_space(dim,1) - leaf_space(dim,0);
          double space_size = space(dim,1) - space(dim,0);
          double ratio = leaf_size / space_size;
          if (ratio > highest_ratio)
          {
            highest_ratio = ratio;
            split_dim = dim;
          }
        }
        std::vector<double> dim_values;
        for (const Eigen::VectorXd & p : leafNode->getPoints())
        {
          dim_values.push_back(p(split_dim));
        }
        split_val = regression_forests::Statistics::median(dim_values);
        break;
      }
      case Type::Random:
      {
        double best_dim_score = 0;
        // For every dimension, try a random split, score it and keep it if necessary
        for (int dim = 0; dim < tree_space.rows(); dim++)
        {
          // choose a random split ensuring there is at least one point on each side
          double s_val_max = std::numeric_limits<double>::lowest();
          double s_val_min = std::numeric_limits<double>::max();
          // Finding min and max points along this dimension
          for (const auto & p : leafNode->getPoints())
          {
            double val = p(dim);
            if (val < s_val_min) s_val_min = val;
            if (val > s_val_max) s_val_max = val; 
          }
          // Choose another dimension if all the points along this dimension are similars
          if (s_val_min == s_val_max) break;
          // Generate random value
          std::uniform_real_distribution<double> val_distrib(s_val_min, s_val_max);
          double curr_split_val = val_distrib(random_engine);
          // This can really happen (even if it not supposed to)
          if (curr_split_val == s_val_max) break;
          // Gathering points
          std::vector<double> values, lower_values, upper_values;
          for (const auto & p : leafNode->getPoints())
          {
            double val = p(dim);
            values.push_back(val);
            if (val <= curr_split_val)
            {
              lower_values.push_back(val);
            }
            else
            {
              upper_values.push_back(val);
            }
          }
          // Size of points set
          int global_size = values.size();
          int lower_size = lower_values.size();
          int upper_size = upper_values.size();
          // variance score [0, 1], 1 is the best
          double global_var = variance(values);
          double lower_var = variance(lower_values);
          double upper_var = variance(upper_values);
          double var_score = (lower_var * lower_size + upper_var * upper_size) / (global_size * global_var);
          var_score = std::max(0.0, 1 - var_score);// Normalization
          // size score [0,1], 1 is the best
          double dim_size = leaf_space(dim,1) - leaf_space(dim,0);
          double lower_ratio = (curr_split_val    - leaf_space(dim,0)) / dim_size;
          double upper_ratio = (leaf_space(dim,1) -    curr_split_val) / dim_size;
          double size_score = 1 - (lower_size * lower_ratio + upper_size * upper_ratio) / (global_size);
          // dim_score [0,1], 1 is the best
          double dim_score = size_score / 2 + var_score / 2;
          if (dim_score > best_dim_score)
          {
            split_dim = dim;
            best_dim_score = dim_score;
            split_val = curr_split_val;
          }
        }
        if (split_dim < 0)
        {
          leafNode->pop_back();
          throw std::runtime_error("No split candidate found");
        }
      }
    }
    leafNode->split(split_dim, split_val);
    nextSplitDim++;
    if (nextSplitDim == leaf_space.rows()) { nextSplitDim = 0;}
  }
  nbPoints++;
}

double MRE::KnownnessTree::getMu() const
{
  int k = tree.dim();
  return 1.0  / floor(std::pow(nbPoints * k / v ,1.0 / k)) * 5;//Dirty hack
}

double MRE::KnownnessTree::getValue(const Eigen::VectorXd& point) const
{
  const kd_trees::KdNode * leaf = tree.getLeaf(point);
  int leaf_count = leaf->getPoints().size();
  Eigen::MatrixXd leaf_space = tree.getSpace(point);
  return getValue(leaf_space, leaf_count);
}

double MRE::KnownnessTree::getValue(const Eigen::MatrixXd& space,
                                    int local_points) const
{
  double max_size = 0;
  const Eigen::MatrixXd & tree_space = tree.getSpace();
  switch(type)
  {
    case Type::Original:
    case Type::Test:
      for (int dim = 0; dim < space.rows(); dim++)
      {
        double local_size = space(dim,1) - space(dim,0);
        double tree_size = tree_space(dim,1) - tree_space(dim,0);
        // Since forall s, norm_inf(s) <= 1, then the length of a dimension is maximum 2
        double size = 2 * local_size / tree_size;
        if (size > max_size)
        {
          max_size = size;
        }
      }
      return std::min(1.0, (double)local_points / v * getMu() / max_size);
    case Type::Random:
    {
      double local_size = 1.0;
      double global_size = 1.0;
      for (int dim = 0; dim < space.rows(); dim++)
      {
        local_size  *= space(dim,1) - space(dim,0);
        global_size *= tree_space(dim,1) - tree_space(dim,0);
      }
      double local_density = local_points / local_size;
      double global_density = nbPoints / global_size;
      double density_ratio = local_density / global_density;
      double raw_value = density_ratio;//std::pow(density_ratio, 1.0 / tree_space.rows());
      double value = std::min(1.0, raw_value);
      return value;
    }
  }
  throw std::runtime_error("Unhandled type for knownness tree");
}

regression_forests::Node * MRE::KnownnessTree::convertToRegNode(const kd_trees::KdNode *node,
                                                                Eigen::MatrixXd &space) const
{
  if (node == NULL) return NULL;
  regression_forests::Node * new_node = new regression_forests::Node();
  // Leaf case
  if (node->isLeaf())
  {
    int nb_points = node->getPoints().size();
    double value = getValue(space, nb_points);
    new_node->a = new regression_forests::PWCApproximation(value);
    return new_node;
  }
  // Node case
  double split_dim = node->getSplitDim();
  double split_val = node->getSplitVal();
  double old_min = space(split_dim, 0);
  double old_max = space(split_dim, 1);
  // Update split
  new_node->s.dim = node->getSplitDim();
  new_node->s.val = node->getSplitVal();
  // Update lower child
  space(split_dim, 1) = split_val;
  new_node->lowerChild = convertToRegNode(node->getLowerChild(), space);
  space(split_dim, 1) = old_max;
  // Update upper child
  space(split_dim, 0) = split_val;
  new_node->upperChild = convertToRegNode(node->getUpperChild(), space);
  space(split_dim, 0) = old_min;
  return new_node;
}

std::unique_ptr<regression_forests::Tree> MRE::KnownnessTree::convertToRegTree() const
{
  std::unique_ptr<regression_forests::Tree> reg_tree(new regression_forests::Tree);
  Eigen::MatrixXd space = tree.getSpace();
  reg_tree->root = convertToRegNode(tree.getRoot(), space);
  return reg_tree;
}

MRE::CustomFPF::CustomFPF(const Eigen::MatrixXd &q_space,
                          int max_points,
                          double reward_max,
                          int nb_trees,
                          KnownnessTree::Type type)
  : r_max(reward_max)
{
  for (int i = 0; i < nb_trees; i++)
  {
    knownness_forest.push_back(MRE::KnownnessTree(q_space, max_points, type));
  }
}

void MRE::CustomFPF::push(const Eigen::VectorXd &q_point)
{
  for (MRE::KnownnessTree &tree : knownness_forest)
  {
    // Adding a point can throw a std::runtime_error in two cases:
    // 1. The point is outside of the tree space (in this case it will be refused by all trees)
    // 2. The random split fails because all points have the same coordinate along the chosen dimension
    // In both case, we choose to 'forget' about this case
    try
    {
      tree.push(q_point);
    }
    catch (const std::runtime_error & exc)
    {
      std::cerr << exc.what() << std::endl;
    }
  }
}

double MRE::CustomFPF::getKnownness(const Eigen::VectorXd& point) const
{
  double sum = 0;
  for (const MRE::KnownnessTree &tree : knownness_forest)
  {
    sum += tree.getValue(point);
  }
  return sum / knownness_forest.size();
}

std::unique_ptr<regression_forests::Forest> MRE::CustomFPF::getKnownnessForest()
{
  std::unique_ptr<regression_forests::Forest> forest(new regression_forests::Forest);
  for (const MRE::KnownnessTree &tree : knownness_forest)
  {
    forest->push(tree.convertToRegTree());
  }
  return forest;
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

std::string to_string(MRE::KnownnessTree::Type type)
{
  switch (type)
  {
    case MRE::KnownnessTree::Type::Original: return "Original";
    case MRE::KnownnessTree::Type::Test: return "Test";
    case MRE::KnownnessTree::Type::Random: return "Random";
  }
  throw std::runtime_error("Unknown type in to_string(MRE::Type)");
}

MRE::KnownnessTree::Type loadType(const std::string &type)
{
  if (type == "Original")
  {
    return MRE::KnownnessTree::Type::Original;
  }
  if (type == "Test")
  {
    return MRE::KnownnessTree::Type::Test;
  }
  if (type == "Random")
  {
    return MRE::KnownnessTree::Type::Random;
  }
  throw std::runtime_error("Unknown KnownnessTree Type: '" + type + "'");
}

}
