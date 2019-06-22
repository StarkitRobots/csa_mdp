#pragma once

#include "starkit_csa_mdp/knownness/knownness_function.h"
#include "kd_trees/kd_tree.h"

#include <starkit_regression_forests/core/tree.h>

#include <starkit_utils/serialization/json_serializable.h>

#include <random>

namespace csa_mdp
{
/// Represents the knownness by using a kd-tree
class KnownnessTree : KnownnessFunction
{
public:
  /// Two types of KnownnessTrees are available
  /// MRE: Follows Multi-Resolution Exploration description (Nouri2009)
  /// Random: Split are performed on random dimensions at a random position
  ///         while maximizing a criteria
  enum class Type
  {
    MRE,
    Random
  };

  /// Config of the KnownnessTree
  class Config : public starkit_utils::JsonSerializable
  {
  public:
    /// Maximal number of points by node (automatically split when values is above)
    int max_points;
    /// Which type is the tree
    Type type;

    Config();

    virtual std::string getClassName() const override;
    virtual Json::Value toJson() const override;
    virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;
  };

  KnownnessTree(const Eigen::MatrixXd& space, const Config& conf);

  double getMu() const;

  // Implementations
  virtual void push(const Eigen::VectorXd& point) override;
  virtual double getValue(const Eigen::VectorXd& point) const override;

  double getValue(const Eigen::MatrixXd& space, int nb_points) const;

  // Conversion tools
  regression_forests::Node* convertToRegNode(const kd_trees::KdNode* node, Eigen::MatrixXd& space) const;
  std::unique_ptr<regression_forests::Tree> convertToRegTree() const;

  /// Ensure that the number of points stored correspond to the total number
  /// of points, if it does not, throw a logic_error
  void checkConsistency();

private:
  /// The basic data structure
  kd_trees::KdTree tree;
  /// Configuration of the tree
  Config conf;
  /// Total number of points
  int nb_points;
  /// On which dimension will happen the next split (MRE type)
  int next_split_dim;
  /// Random generator (for Random Type)
  std::default_random_engine random_engine;
};

std::string to_string(KnownnessTree::Type type);
KnownnessTree::Type loadType(const std::string& type);

}  // namespace csa_mdp
