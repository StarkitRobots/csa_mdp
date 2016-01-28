#include "kd_trees/kd_node.h"

namespace kd_trees
{

KdNode::KdNode()
  : lChild(NULL), uChild(NULL), splitDim(-1), splitValue(0.0)
{
}

bool KdNode::isLeaf() const
{
  return lChild == NULL;
}

KdNode * KdNode::getLeaf(const Eigen::VectorXd& point)
{
  if (isLeaf()) {
    return this;
  }
  if (point(splitDim) > splitValue) {
    return uChild->getLeaf(point);
  }
  return lChild->getLeaf(point);
}

const KdNode * KdNode::getLeaf(const Eigen::VectorXd& point) const
{
  if (isLeaf()) {
    return this;
  }
  if (point(splitDim) > splitValue) {
    return uChild;
  }
  return lChild;
}

const KdNode * KdNode::getLowerChild() const
{
  return lChild;
}

const KdNode * KdNode::getUpperChild() const
{
  return uChild;
}

int KdNode::getSplitDim() const
{
  return splitDim;
}

double KdNode::getSplitVal() const
{
  return splitValue;
}

void KdNode::push(const Eigen::VectorXd& point)
{
  points.push_back(point);
}

void KdNode::pop_back()
{
  points.pop_back();
}

void KdNode::split(int dim, double value)
{
  if (!isLeaf()) {
    throw std::runtime_error("KdNode: Cannot split a non-leaf node");
  }
  splitDim = dim;
  splitValue = value;
  lChild = new KdNode();
  uChild = new KdNode();
  for (const auto & p : points) {
    if (p(splitDim) > splitValue) {
      uChild->push(p);
    }
    else {
      lChild->push(p);
    }
  }
}

void KdNode::leafSpace(Eigen::MatrixXd &space, const Eigen::VectorXd &point) const
{
  if (isLeaf()) {
    return;
  }
  if (point(splitDim) > splitValue) {
    space(splitDim,0) = splitValue;
    uChild->leafSpace(space, point);
  }
  else {
    space(splitDim,1) = splitValue;
    lChild->leafSpace(space, point);
  }
}

const std::vector<Eigen::VectorXd>& KdNode::getPoints() const
{
  return points;
}

}
