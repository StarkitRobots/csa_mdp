#pragma once

#include <Eigen/Core>

#include <vector>

namespace kd_trees
{

class KdNode {
private:
  KdNode * lChild;// point(splitDim) <= splitValue
  KdNode * uChild;// point(splitDim)  > splitValue
  int splitDim;
  double splitValue;
  std::vector<Eigen::VectorXd> points;

public:
  KdNode();

  bool isLeaf() const;

  // Get the leaf corresponding to the given point
  KdNode * getLeaf(const Eigen::VectorXd &point);
  const KdNode * getLeaf(const Eigen::VectorXd &point) const;

  const KdNode * getLowerChild() const;
  const KdNode * getUpperChild() const;
  int getSplitDim() const;
  double getSplitVal() const;
  

  // Add the point to the current tree
  void push(const Eigen::VectorXd &point);

  // Split node and add separate points to his child
  void split(int splitDim, double splitValue);

  // Update the given space to match the space of the leaf concerning the
  // provided point space is a N by 2 matrix where space(d,0) is the min and
  // space(d,1) is the max
  void leafSpace(Eigen::MatrixXd &space, const Eigen::VectorXd &point) const;

  const std::vector<Eigen::VectorXd> &getPoints() const;
};

}
