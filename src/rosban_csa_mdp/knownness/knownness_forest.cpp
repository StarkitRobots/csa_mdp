#include "rosban_csa_mdp/knownness/knownness_forest.h"

namespace csa_mdp
{

KnownnessForest::Config::Config()
  : nb_trees(25), tree_conf()
{
}

std::string KnownnessForest::Config::class_name() const
{
  return "KnownnessForestConfig";
}

void KnownnessForest::Config::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("nb_trees", nb_trees, out);
  tree_conf.write("tree_conf", out);
}

void KnownnessForest::Config::from_xml(TiXmlNode *node)
{
  nb_trees = rosban_utils::xml_tools::read<int>(node, "nb_trees");
  tree_conf.tryRead(node, "tree_conf");
}

KnownnessForest::KnownnessForest()
{
}

KnownnessForest::KnownnessForest(const Eigen::MatrixXd &space,
                                 const Config &conf)
{
  for (int tree = 0; tree < conf.nb_trees; tree++)
  {
    trees.push_back(KnownnessTree(space, conf.tree_conf));
  }
}

void KnownnessForest::push(const Eigen::VectorXd &point)
{
  for (KnownnessTree &tree : trees)
  {
    // Adding a point can throw a std::runtime_error in two cases:
    // 1. The point is outside of the tree space (in this case it will be refused by all trees)
    // 2. The random split fails because all points have the same coordinate along the chosen dimension
    // In both case, we choose to 'forget' about this case
    try
    {
      tree.push(point);
    }
    catch (const std::runtime_error & exc)
    {
      std::cerr << exc.what() << std::endl;
    }
  }
}

/// Get the knownness value at the given point
double KnownnessForest::getValue(const Eigen::VectorXd &point) const
{
  double sum = 0;
  for (const KnownnessTree &tree : trees)
  {
    sum += tree.getValue(point);
  }
  return sum / trees.size();
}

std::unique_ptr<regression_forests::Forest> KnownnessForest::convertToRegressionForest() const
{
  std::unique_ptr<regression_forests::Forest> forest(new regression_forests::Forest);
  for (const KnownnessTree &tree : trees)
  {
    forest->push(tree.convertToRegTree());
  }
  return forest;
}

}
