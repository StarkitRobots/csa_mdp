#include "rosban_csa_mdp/solvers/extra_trees.h"

#include <iostream>

using regression_forests::ApproximationType;
using regression_forests::RandomizedTrees;
using regression_forests::TrainingSet;

namespace csa_mdp
{

FQI::Config::Config()
{
  horizon = 1;
  discount = 0.99;
  max_action_tiles = 0;
  time = 0;
}

std::vector<std::string> FQI::Config::names() const
{
  std::vector<std::string> result = {"Horizon"       ,
                                     "Discount"      ,
                                     "MaxActionTiles",
                                     "Time"};
  std::vector<std::string> etNames = q_learning_conf.names();
  result.insert(result.end(), etNames.begin(), etNames.end());
  return result;
}

std::vector<std::string> FQI::Config::values() const
{
  std::vector<std::string> result;
  result.push_back(std::to_string(horizon       ));
  result.push_back(std::to_string(discount      ));
  result.push_back(std::to_string(max_action_tiles));
  result.push_back(std::to_string(time          ));
  std::vector<std::string> etValues = q_learning_conf.values();
  result.insert(result.end(), etValues.begin(), etValues.end());
  return result;
}

void FQI::Config::load(const std::vector<std::string>& colNames,
                       const std::vector<std::string>& colValues)
{
  auto expectedNames = names();
  if (colNames.size() != expectedNames.size()) {
    throw std::runtime_error("Failed to load extraTreesConfig, mismatch of vector size");
  }
  for (size_t colNo = 0;  colNo < colNames.size(); colNo++) {
    auto givenName = colNames[colNo];
    auto expectedName = expectedNames[colNo];
    if (givenName.find(expectedName) == std::string::npos) {
      throw std::runtime_error("Given name '" + givenName + "' does not match '"
                               + expectedName + "'");
    }
  }
  horizon        = std::stoi(colValues[0]);
  discount       = std::stoi(colValues[1]);
  max_action_tiles = std::stoi(colValues[2]);
  time           = std::stod(colValues[3]);
  std::vector<std::string> etNames, etValues;
  etNames.insert (etNames.end() , colNames.begin() + 4 , colNames.end() );
  etValues.insert(etValues.end(), colValues.begin() + 4, colValues.end());
  q_learning_conf.load(etNames, etValues);
}


FQI::FQI(const Eigen::MatrixXd& xLimits_,
                       const Eigen::MatrixXd& uLimits_)
  : xLimits(xLimits_), uLimits(uLimits_)
{
  xDim = xLimits.rows();
  uDim = uLimits.rows();
}

const regression_forests::Forest& FQI::valueForest()
{
  return *q_value;
}

void FQI::solve(const std::vector<Sample>& samples,
                size_t horizon, double discount,
                std::function<bool(const Eigen::VectorXd&)> isTerminal,
                size_t k, size_t nmin, size_t nbTrees,
                double minVariance,
                bool bootstrap, bool preFilter, bool parallelMerge,
                ApproximationType apprType)
{
  q_value.release();
  for (size_t h = 1; h <= horizon; h++) {
    // Compute learningSet with last q_value
    TrainingSet ls = getTrainingSet(samples, discount, isTerminal, preFilter, parallelMerge);
    // Compute q_value from learningSet
    q_value = RandomizedTrees::extraTrees(ls, k, nmin, nbTrees, minVariance, bootstrap, apprType);
  }
}

void FQI::solve(const std::vector<Sample>& samples,
                Config& conf,
                std::function<bool(const Eigen::VectorXd&)> isTerminal)
{
  solve(samples,
        conf.horizon, conf.discount,
        isTerminal,
        conf.q_learning_conf.k, conf.q_learning_conf.nMin, conf.q_learning_conf.nbTrees,
        conf.q_learning_conf.minVar,
        conf.q_learning_conf.bootstrap, conf.preFilter, conf.parallelMerge,
        conf.q_learning_conf.apprType);
}

TrainingSet
FQI::getTrainingSet(const std::vector<Sample>& samples,
                    double discount,
                    std::function<bool(const Eigen::VectorXd&)> isTerminal,
                    bool preFilter, bool parallelMerge)
{
  TrainingSet ls(xDim + uDim);
  for (size_t i = 0; i < samples.size(); i++) {
    const Sample& sample = samples[i];
    int xDim = sample.state.rows();
    int uDim = sample.action.rows();
    Eigen::VectorXd input(xDim + uDim);
    input.segment(0, xDim) = sample.state;
    input.segment(xDim, uDim) = sample.action;
    Eigen::VectorXd nextState = sample.next_state;
    double reward = sample.reward;
    if (q_value && !isTerminal(nextState)) {
      // Establishing limits for projection
      Eigen::MatrixXd limits(xDim + uDim, 2);
      limits.block(0, 0, xDim, 1) = nextState;
      limits.block(0, 1, xDim, 1) = nextState;
      limits.block(xDim, 0, uDim, 2) = uLimits;
      std::unique_ptr<regression_forests::Tree> subTree;
      subTree = q_value->unifiedProjectedTree(limits, max_action_tiles, preFilter, parallelMerge);
      reward += discount * subTree->getMax(limits);
    }
    ls.push(regression_forests::Sample(input, reward));
  }
  return ls;
}

}
