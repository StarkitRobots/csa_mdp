#pragma once

#include "rosban_csa_mdp/core/sample.h"

#include "rosban_regression_forests/core/training_set.h"
#include "rosban_regression_forests/algorithms/randomized_tree.h"

namespace csa_mdp
{

class FQI {
public:
  class Config{
  public:
    size_t horizon;
    double discount;
    size_t max_action_tiles;//Max action tiles when computing optimal action
    double time;
    regression_forests::RandomizedTrees::Config q_learning_conf;
    regression_forests::RandomizedTrees::Config policy_learning_conf;

    Config();
    std::vector<std::string> names() const;
    std::vector<std::string> values() const;
    void load(const std::vector<std::string>& names,
              const std::vector<std::string>& values);
  };

private:
  std::unique_ptr<regression_forests::Forest> q_value;
  Eigen::MatrixXd xLimits;
  Eigen::MatrixXd uLimits;
  size_t xDim;
  size_t uDim;

public:
  Config conf;

  FQI(const Eigen::MatrixXd& xLimits,
      const Eigen::MatrixXd& uLimits);

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
   * Provide a training set using q_value
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
