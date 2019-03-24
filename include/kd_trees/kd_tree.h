#pragma once

#include "kd_trees/kd_node.h"

namespace kd_trees
{
class KdTree
{
private:
  KdNode root;
  Eigen::MatrixXd space;

public:
  KdTree(const Eigen::MatrixXd& space);

  int dim() const;
  const KdNode* getRoot() const;

  /// Return all the leaves inside the tree
  std::vector<KdNode*> getLeaves();

  KdNode* getLeaf(const Eigen::VectorXd& point);
  const KdNode* getLeaf(const Eigen::VectorXd& point) const;

  void push(const Eigen::VectorXd& point);

  const Eigen::MatrixXd& getSpace() const;
  Eigen::MatrixXd getSpace(const Eigen::VectorXd& point) const;
};

}  // namespace kd_trees
