#pragma once

#include "starkit_csa_mdp/knownness/knownness_tree.h"

#include "starkit_regression_forests/core/forest.h"

namespace csa_mdp
{
class KnownnessForest : public KnownnessFunction
{
public:
  class Config : public starkit_utils::JsonSerializable
  {
  public:
    Config();

    virtual std::string getClassName() const override;
    virtual Json::Value toJson() const override;
    virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

    int nb_trees;
    KnownnessTree::Config tree_conf;
  };

  KnownnessForest();
  KnownnessForest(const Eigen::MatrixXd& space, const Config& conf);

  /// Notify the knownness function that a new point has been found
  virtual void push(const Eigen::VectorXd& point);

  /// Get the knownness value at the given point
  virtual double getValue(const Eigen::VectorXd& point) const;

  /// Conversion to a regression_forest
  std::unique_ptr<regression_forests::Forest> convertToRegressionForest() const;

  /// Ensure that all the trees are consistent
  void checkConsistency();

private:
  std::vector<KnownnessTree> trees;
};

}  // namespace csa_mdp
