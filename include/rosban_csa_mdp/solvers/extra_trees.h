#pragma once

#include "rosban_csa_mdp/core/sample.h"

#include "rosban_regression_forests/core/training_set.h"
#include "rosban_regression_forests/algorithms/randomized_tree.h"

namespace csa_mdp
{

class ExtraTrees {
public:
  class Config{
  public:
    size_t horizon;
    double discount;
    bool preFilter;
    bool parallelMerge;
    size_t maxActionTiles;
    double time;
    regression_forests::RandomizedTrees::Config ETConf;

    Config();
    std::vector<std::string> names() const;
    std::vector<std::string> values() const;
    void load(const std::vector<std::string>& names,
              const std::vector<std::string>& values);
  };

private:
  std::unique_ptr<regression_forests::Forest> qValue;
  Eigen::MatrixXd xLimits;
  Eigen::MatrixXd uLimits;
  size_t xDim;
  size_t uDim;
  size_t maxActionTiles;
public:
  ExtraTrees(const Eigen::MatrixXd& xLimits,
             const Eigen::MatrixXd& uLimits,
             size_t maxActionTiles = 0);

  const regression_forests::Forest& valueForest();

  void solve(const std::vector<Sample>& samples,
             size_t horizon, double discount,
             std::function<bool(const Eigen::VectorXd&)> isTerminal,
             size_t k, size_t nmin, size_t nbTrees, double minVariance,
             bool bootstrap, bool preFilter = false, bool parallelMerge = false,
             regression_forests::ApproximationType apprType =
             regression_forests::ApproximationType::PWC);

  void solve(const std::vector<Sample>& samples,
             Config& conf,
             std::function<bool(const Eigen::VectorXd&)> isTerminal);

  /**
   * Provide a training set using qValue
   */
  regression_forests::TrainingSet
  getTrainingSet(const std::vector<Sample>& samples,
                 double discount,
                 std::function<bool(const Eigen::VectorXd&)> isTerminal,
                 bool preFilter, bool parallelMerge);

  static Eigen::VectorXd
  bestAction(std::unique_ptr<regression_forests::Forest> f,
             const Eigen::VectorXd& state);
};


}
