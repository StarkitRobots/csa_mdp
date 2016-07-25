#include "rosban_csa_mdp/reward_predictors/monte_carlo_predictor.h"

#include "rosban_regression_forests/tools/statistics.h"

#include "rosban_random/tools.h"

namespace csa_mdp
{

MonteCarloPredictor::MonteCarloPredictor()
  : nb_predictions(100)
{
  engine = rosban_random::getRandomEngine();
}

void MonteCarloPredictor::predict(const Eigen::VectorXd & input,
                                  std::shared_ptr<const Policy> policy,
                                  int nb_steps,
                                  std::shared_ptr<Problem> model,//TODO: Model class ?
                                  RewardFunction reward_function,
                                  double discount,
                                  double * mean,
                                  double * var)
{
  std::vector<double> rewards;
  rewards.reserve(nb_predictions);
  for (int prediction = 0; prediction < nb_predictions; prediction++)
  {
    double coeff = 1;
    double reward = 0;
    Eigen::VectorXd state = input;
    for (int i = 0; i < nb_steps; i++)
    {
      Eigen::VectorXd action = policy->getAction(state, &engine);
      Eigen::VectorXd next_state = model->getSuccessor(state, action);
      reward += coeff * reward_function(state, action, next_state);
      state = next_state;
      coeff *= discount;
    }
    rewards.push_back(reward);
  }
  double internal_mean = regression_forests::Statistics::mean(rewards);
  if (mean != nullptr) *mean = internal_mean;
  if (var != nullptr) *var = regression_forests::Statistics::variance(rewards);
}

std::string MonteCarloPredictor::class_name() const
{
  return "MonteCarloPredictor";
}

void MonteCarloPredictor::to_xml(std::ostream & out) const
{
  rosban_utils::xml_tools::write<int>("nb_predictions", nb_predictions, out);
}

void MonteCarloPredictor::from_xml(TiXmlNode * node)
{
  rosban_utils::xml_tools::try_read<int>(node, "nb_predictions", nb_predictions);
}

}
