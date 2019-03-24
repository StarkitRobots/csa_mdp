#include "rhoban_csa_mdp/knownness/knownness_forest.h"

#include <iostream>

namespace csa_mdp
{
KnownnessForest::Config::Config() : nb_trees(25), tree_conf()
{
}

std::string KnownnessForest::Config::getClassName() const
{
  return "KnownnessForestConfig";
}

Json::Value KnownnessForest::Config::toJson() const
{
  Json::Value v;
  v["nb_trees"] = nb_trees;
  v["tree_conf"] = tree_conf.toJson();
  return v;
}

void KnownnessForest::Config::fromJson(const Json::Value& v, const std::string& dir_name)
{
  nb_trees = rhoban_utils::read<int>(v, "nb_trees");
  tree_conf.tryRead(v, "tree_conf", dir_name);
}

KnownnessForest::KnownnessForest()
{
}

KnownnessForest::KnownnessForest(const Eigen::MatrixXd& space, const Config& conf)
{
  for (int tree = 0; tree < conf.nb_trees; tree++)
  {
    trees.push_back(KnownnessTree(space, conf.tree_conf));
  }
}

void KnownnessForest::push(const Eigen::VectorXd& point)
{
  for (KnownnessTree& tree : trees)
  {
    // Adding a point can throw a std::runtime_error in two cases:
    // 1. The point is outside of the tree space (in this case it will be refused by all trees)
    // 2. The random split fails because all points have the same coordinate along the chosen dimension
    // In both case, we choose to 'forget' about this case
    try
    {
      tree.push(point);
    }
    catch (const std::runtime_error& exc)
    {
      std::cerr << exc.what() << std::endl;
    }
  }
}

/// Get the knownness value at the given point
double KnownnessForest::getValue(const Eigen::VectorXd& point) const
{
  double sum = 0;
  for (const KnownnessTree& tree : trees)
  {
    double tree_value = tree.getValue(point);
    sum += tree_value;
  }
  return sum / trees.size();
}

std::unique_ptr<regression_forests::Forest> KnownnessForest::convertToRegressionForest() const
{
  std::unique_ptr<regression_forests::Forest> forest(new regression_forests::Forest);
  for (const KnownnessTree& tree : trees)
  {
    forest->push(tree.convertToRegTree());
  }
  return forest;
}

void KnownnessForest::checkConsistency()
{
  for (KnownnessTree& tree : trees)
  {
    tree.checkConsistency();
  }
}

}  // namespace csa_mdp
