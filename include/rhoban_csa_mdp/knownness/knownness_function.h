#pragma once

#include <Eigen/Core>

namespace csa_mdp
{

/// Knownness functions are function approximator from R^n to [0,1]
/// Knownness of 0 is minimal
/// Knownness of 1 is maximal
class KnownnessFunction
{
public:

  /// Notify the knownness function that a new point has been found
  virtual void push(const Eigen::VectorXd &point) = 0;

  /// Get the knownness value at the given point
  virtual double getValue(const Eigen::VectorXd &point) const = 0;
};

}
