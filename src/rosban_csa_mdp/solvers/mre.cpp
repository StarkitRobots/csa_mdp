#include "rosban_csa_mdp/solvers/mre.h"

#include "rosban_regression_forests/approximations/pwc_approximation.h"
#include "rosban_regression_forests/tools/random.h"

#include <iostream>

using regression_forests::TrainingSet;

namespace csa_mdp
{

MRE::KnownnessTree::KnownnessTree(const Eigen::MatrixXd& space, int maxPoints)
  : tree(space), v(maxPoints), nextSplitDim(0), nbPoints(0)
{
}

void MRE::KnownnessTree::push(const Eigen::VectorXd& point)
{
  kd_trees::KdNode * leafNode = tree.getLeaf(point);
  Eigen::MatrixXd leafSpace = tree.getSpace(point);
  leafNode->push(point);
  int leafCount = leafNode->getPoints().size();
  if (leafCount > v) {
    int split_dim = nextSplitDim;
    //// Hack, splitting on the biggest dimension
    //double max_size = 0;
    //split_dim = -1;
    //for (int dim = 0; dim < leafSpace.rows(); dim++)
    //{
    //  double dim_size = leafSpace(dim,1) - leafSpace(dim,0);
    //  if (dim_size > max_size)
    //  {
    //    max_size = dim_size;
    //    split_dim = dim;
    //  }
    //}
    //// End of hack
    double split_val = (leafSpace(split_dim, 0) + leafSpace(split_dim,1)) / 2;
    leafNode->split(split_dim, split_val);
    //std::cerr << "Splitting at: (" << split_dim << "," << split_val << ")" << std::endl;
    nextSplitDim++;
    if (nextSplitDim == leafSpace.rows()) { nextSplitDim = 0;}
  }
  nbPoints++;
}

double MRE::KnownnessTree::getMu() const
{
  int k = tree.dim();
  return 1.0  / floor(std::pow(nbPoints * k / v ,1.0 / k));
}

double MRE::KnownnessTree::getValue(const Eigen::VectorXd& point) const
{
  const kd_trees::KdNode * leaf = tree.getLeaf(point);
  int leaf_count = leaf->getPoints().size();
  Eigen::MatrixXd leaf_space = tree.getSpace(point);
  return getValue(leaf_space, leaf_count);
}

double MRE::KnownnessTree::getValue(const Eigen::MatrixXd& space,
                                    int nb_points) const
{
  Eigen::VectorXd space_sizes = space.block(0, 1, space.rows(), 1) - space.block(0, 0, space.rows(), 1);
  double size = space_sizes.maxCoeff();
  return std::min(1.0, (double)nb_points / v * getMu() / size);
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
    new_node->a = new regression_forests::PWCApproximation(getValue(space, nb_points));
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
                          double reward_max)
  : knownness_tree(q_space, max_points),
    r_max(reward_max)
{
}

void MRE::CustomFPF::push(const Eigen::VectorXd &q_point)
{
  knownness_tree.push(q_point);
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
    double knownness = knownness_tree.getValue(input);
    double new_reward = reward * knownness + r_max * (1 - knownness);
    new_ts.push(regression_forests::Sample(input, new_reward));
  }
  return new_ts;
}

MRE::MRE(const Eigen::MatrixXd &state_space_,
         const Eigen::MatrixXd &action_space_,
         int max_points,
         double reward_max,
         int plan_period_,
         const FPF::Config &fpf_conf,
         std::function<bool(const Eigen::VectorXd &)> is_terminal_)
  : plan_period(plan_period_),
    is_terminal(is_terminal_),
    solver(Eigen::MatrixXd(0,0),0,0),
    state_space(state_space_),
    action_space(action_space_)
{
  int s_dim = state_space.rows();
  int a_dim = action_space.rows();
  Eigen::MatrixXd q_space(s_dim + a_dim, 2);
  q_space.block(    0, 0, s_dim, 2) = state_space;
  q_space.block(s_dim, 0, a_dim, 2) = action_space;
  solver = CustomFPF(q_space, max_points, reward_max);
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
  std::unique_ptr<regression_forests::Forest> forest(new regression_forests::Forest);
  forest->push(solver.knownness_tree.convertToRegTree());
  forest->save(prefix + "knownness.data");
}

void MRE::saveStatus(const std::string &prefix)
{
  savePolicies(prefix);
  saveValue(prefix);
  saveKnownnessTree(prefix);
}

}
