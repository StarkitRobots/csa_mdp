#include "rosban_csa_mdp/solvers/extra_trees.h"

#include <iostream>

using regression_forests::ApproximationType;
using regression_forests::RandomizedTrees;
using regression_forests::TrainingSet;

namespace csa_mdp
{

ExtraTrees::Config::Config()
{
  horizon = 1;
  discount = 0.99;
  preFilter = false;
  parallelMerge = true;
  maxActionTiles = 0;
  time = 0;
}

std::vector<std::string> ExtraTrees::Config::names() const
{
  std::vector<std::string> result = {"Horizon"       ,
                                     "Discount"      ,
                                     "PreFilter"     ,
                                     "ParallelMerge" ,
                                     "MaxActionTiles",
                                     "Time"};
  std::vector<std::string> etNames = ETConf.names();
  result.insert(result.end(), etNames.begin(), etNames.end());
  return result;
}

std::vector<std::string> ExtraTrees::Config::values() const
{
  std::vector<std::string> result;
  result.push_back(std::to_string(horizon       ));
  result.push_back(std::to_string(discount      ));
  result.push_back(std::to_string(preFilter     ));
  result.push_back(std::to_string(parallelMerge ));
  result.push_back(std::to_string(maxActionTiles));
  result.push_back(std::to_string(time          ));
  std::vector<std::string> etValues = ETConf.values();
  result.insert(result.end(), etValues.begin(), etValues.end());
  return result;
}

void ExtraTrees::Config::load(const std::vector<std::string>& colNames,
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
  preFilter      = std::stoi(colValues[2]);
  parallelMerge  = std::stoi(colValues[3]);
  maxActionTiles = std::stoi(colValues[4]);
  time           = std::stod(colValues[5]);
  std::vector<std::string> etNames, etValues;
  etNames.insert (etNames.end() , colNames.begin() + 6 , colNames.end() );
  etValues.insert(etValues.end(), colValues.begin() + 6, colValues.end());
  ETConf.load(etNames, etValues);
}


ExtraTrees::ExtraTrees(const Eigen::MatrixXd& xLimits_,
                       const Eigen::MatrixXd& uLimits_,
                       size_t maxActionTiles_)
  : xLimits(xLimits_), uLimits(uLimits_), maxActionTiles(maxActionTiles_)
{
  xDim = xLimits.rows();
  uDim = uLimits.rows();
}

const regression_forests::Forest& ExtraTrees::valueForest()
{
  return *qValue;
}

void ExtraTrees::solve(const std::vector<Sample>& samples,
                       size_t horizon, double discount,
                       std::function<bool(const Eigen::VectorXd&)> isTerminal,
                       size_t k, size_t nmin, size_t nbTrees,
                       double minVariance,
                       bool bootstrap, bool preFilter, bool parallelMerge,
                       ApproximationType apprType)
{
  qValue.release();
  for (size_t h = 1; h <= horizon; h++) {
    // Compute learningSet with last qValue
    TrainingSet ls = getTrainingSet(samples, discount, isTerminal, preFilter, parallelMerge);
    // Compute qValue from learningSet
    qValue = RandomizedTrees::extraTrees(ls, k, nmin, nbTrees, minVariance, bootstrap, apprType);
  }
}

void ExtraTrees::solve(const std::vector<Sample>& samples,
                       Config& conf,
                       std::function<bool(const Eigen::VectorXd&)> isTerminal)
{
  solve(samples,
        conf.horizon, conf.discount,
        isTerminal,
        conf.ETConf.k, conf.ETConf.nMin, conf.ETConf.nbTrees,
        conf.ETConf.minVar,
        conf.ETConf.bootstrap, conf.preFilter, conf.parallelMerge,
        conf.ETConf.apprType);
}

TrainingSet
ExtraTrees::getTrainingSet(const std::vector<Sample>& samples,
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
    if (qValue && !isTerminal(nextState)) {
      // Establishing limits for projection
      Eigen::MatrixXd limits(xDim + uDim, 2);
      limits.block(0, 0, xDim, 1) = nextState;
      limits.block(0, 1, xDim, 1) = nextState;
      limits.block(xDim, 0, uDim, 2) = uLimits;
      std::unique_ptr<regression_forests::Tree> subTree;
      subTree = qValue->unifiedProjectedTree(limits, maxActionTiles, preFilter, parallelMerge);
      reward += discount * subTree->getMax(limits);
    }
    ls.push(regression_forests::Sample(input, reward));
  }
  return ls;
}

}
