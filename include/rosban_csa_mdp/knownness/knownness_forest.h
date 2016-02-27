#pragma once

#include "rosban_csa_mdp/knownness/knownness_tree.h"

#include "rosban_regression_forests/core/forest.h"

namespace csa_mdp
{

class KnownnessForest : public KnownnessFunction
{
public:
  class Config : public rosban_utils::Serializable
  {
  public:

    Config();

    virtual std::string class_name() const override;
    virtual void to_xml(std::ostream &out) const override;
    virtual void from_xml(TiXmlNode *node) override;

    int nb_trees;
    KnownnessTree::Config tree_conf;
  };

  KnownnessForest();
  KnownnessForest(const Eigen::MatrixXd &space,
                  const Config &conf);

  /// Notify the knownness function that a new point has been found
  virtual void push(const Eigen::VectorXd &point);

  /// Get the knownness value at the given point
  virtual double getValue(const Eigen::VectorXd &point) const;

  /// Conversion to a regression_forest
  std::unique_ptr<regression_forests::Forest> convertToRegressionForest() const;

private:
  std::vector<KnownnessTree> trees;
};

}
