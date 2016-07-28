#pragma once

#include "rosban_csa_mdp/reward_predictors/reward_predictor.h"

#include <random>

namespace csa_mdp
{

class MonteCarloPredictor : public RewardPredictor
{
public:
  MonteCarloPredictor();

  void predict(const Eigen::VectorXd & input,
               std::shared_ptr<const Policy> policy,
               std::shared_ptr<Problem> model,//TODO: Model class ?
               Problem::RewardFunction reward_function,
               Problem::ValueFunction value_function,
               Problem::TerminalFunction terminal_function,
               double discount,
               double * mean,
               double * var,
               std::default_random_engine * engine) override;

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

private:
  /// Number of trajectories simulated
  int nb_predictions;

  /// Number of steps before using ValueFunction
  int nb_steps;

  //TODO add number of threads
  std::default_random_engine engine;
};

}
