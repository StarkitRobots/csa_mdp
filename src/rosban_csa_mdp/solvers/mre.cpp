#include "rosban_csa_mdp/solvers/mre.h"

#include "rosban_regression_forests/approximations/pwc_approximation.h"
#include "rosban_regression_forests/tools/random.h"
#include "rosban_regression_forests/tools/statistics.h"

#include <iostream>

using regression_forests::TrainingSet;

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
  const Eigen::MatrixXd &tree_space = tree.getSpace(point);
  for (int dim = 0; dim < point.rows(); dim++)
  {
    if (point(dim) < tree_space(dim,0) || point(dim) > tree_space(dim,1))
    {
      throw std::runtime_error("Point is outside of space!");
    }
  }
  // Pushing point
  kd_trees::KdNode * leafNode = tree.getLeaf(point);
  Eigen::MatrixXd leafSpace = tree.getSpace(point);
  leafNode->push(point);
  int leafCount = leafNode->getPoints().size();
  if (leafCount > v) {
    int split_dim;
    double split_val;
    switch(type)
    {
      case Original:
        split_dim = nextSplitDim;
        split_val = (leafSpace(split_dim, 0) + leafSpace(split_dim,1)) / 2;
        break;
      case Test:
      {
        const Eigen::MatrixXd &space = tree.getSpace();
        double highest_ratio = 0;
        split_dim = -1;
        for (int dim = 0; dim < leafSpace.rows(); dim++)
        {
          double leaf_size = leafSpace(dim,1) - leafSpace(dim,0);
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
      case Random:
      {
        std::uniform_int_distribution<int> dim_distrib(0, leafSpace.rows() - 1);
        split_dim = dim_distrib(random_engine);
        double s_val_max = std::numeric_limits<double>::lowest();
        double s_val_min = std::numeric_limits<double>::max();
        for (const auto & p : leafNode->getPoints())
        {
          double val = p(split_dim);
          if (val < s_val_min) s_val_min = val;
          if (val > s_val_max) s_val_max = val; 
        }
        if (s_val_min == s_val_max)
        {
          throw std::runtime_error("All values are the same, cannot operate a random split");
        }
        std::uniform_real_distribution<double> val_distrib(s_val_min, s_val_max);
        split_val = val_distrib(random_engine);
        // This can really happen
        if (split_val == s_val_max)
        {
          throw std::runtime_error("s_val_max == split_val, forbidden situation");
        }
      }
    }
    leafNode->split(split_dim, split_val);
    nextSplitDim++;
    if (nextSplitDim == leafSpace.rows()) { nextSplitDim = 0;}
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
    case Original:
    case Test:
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
    case Random:
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
      double value = std::min(1.0, local_density / global_density);
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
  TrainingSet original_ts = FPF::getTrainingSet(samples, is_terminal);
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

std::vector<Eigen::VectorXd> MRE::CustomFPF::getPolicyTrainingStates(const std::vector<Sample>& samples)
{
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples.size());
  for (const Sample & s : samples)
  {
    result.push_back(s.state);
  }
  return result;
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
