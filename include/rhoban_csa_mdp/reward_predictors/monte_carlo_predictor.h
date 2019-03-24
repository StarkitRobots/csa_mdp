#pragma once

#include "rhoban_csa_mdp/reward_predictors/reward_predictor.h"

#include <random>

namespace csa_mdp
{
class MonteCarloPredictor : public RewardPredictor
{
public:
  MonteCarloPredictor();

  void predict(const Eigen::VectorXd& input, const Policy& policy, Problem::ResultFunction result_function,
               Problem::ValueFunction value_function, double discount, double* mean, double* var,
               std::default_random_engine* engine) override;

  typedef std::function<void(int start, int end, std::default_random_engine* engine)> RPTask;

  virtual RPTask getTask(const Eigen::VectorXd& input, const Policy& policy, Problem::ResultFunction result_function,
                         Problem::ValueFunction value_function, double discount, std::vector<double>& rewards);

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

private:
  /// Number of trajectories simulated
  int nb_predictions;

  /// Number of steps before using ValueFunction
  int nb_steps;
};

}  // namespace csa_mdp
