#include "rosban_csa_mdp/solvers/mre.h"

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

double MRE::KnownnessTree::getValue(const Eigen::MatrixXd& point) const
{
  const kd_trees::KdNode * leaf = tree.getLeaf(point);
  int leafCount = leaf->getPoints().size();
  Eigen::MatrixXd leafSpace = tree.getSpace(point);
  Eigen::VectorXd space_sizes = leafSpace.block(0, 1, leafSpace.rows(), 1) - leafSpace.block(0, 0, leafSpace.rows(), 1);
  double leafSize = space_sizes.maxCoeff();
  return std::min(1.0, leafCount / v * getMu() / leafSize);
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
  if (samples.size() % plan_period == 0)
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

}
