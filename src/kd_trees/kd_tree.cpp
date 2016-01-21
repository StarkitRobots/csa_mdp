#include "kd_trees/kd_tree.h"

namespace kd_trees
{

KdTree::KdTree(const Eigen::MatrixXd & tree_space)
  : space(tree_space)
{
}

const KdNode * KdTree::getRoot() const
{
  return &root;
}

int KdTree::dim() const
{
  return space.rows();
}

KdNode * KdTree::getLeaf(const Eigen::VectorXd& point)
{
  return root.getLeaf(point);
}

const KdNode * KdTree::getLeaf(const Eigen::VectorXd& point) const
{
  return root.getLeaf(point);
}

void KdTree::push(const Eigen::VectorXd& point)
{
  getLeaf(point)->push(point);
}

const Eigen::MatrixXd & KdTree::getSpace() const
{
  return space;
}

Eigen::MatrixXd KdTree::getSpace(const Eigen::VectorXd& point) const
{
  Eigen::MatrixXd leaf_space = space;
  root.leafSpace(leaf_space, point);
  return leaf_space;
}

}
