#pragma once

#include "kd_trees/kd_node.h"

namespace Math {
  namespace KdTrees {

    class KdTree {
    private:
      KdNode root;
      Eigen::MatrixXd space;

    public:
      KdTree(const Eigen::MatrixXd &space);

      int dim() const;

      KdNode * getLeaf(const Eigen::VectorXd &point);

      void push(const Eigen::VectorXd &point);

      Eigen::MatrixXd getSpace(const Eigen::VectorXd &point) const;
    };
  }
}
