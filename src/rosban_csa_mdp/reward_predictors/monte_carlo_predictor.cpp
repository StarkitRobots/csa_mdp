#include "rosban_csa_mdp/reward_predictors/monte_carlo_predictor.h"

#include "rosban_regression_forests/tools/statistics.h"

#include "rosban_random/tools.h"

#include "rosban_utils/multi_core.h"

using rosban_utils::MultiCore;

namespace csa_mdp
{

MonteCarloPredictor::MonteCarloPredictor()
  : nb_predictions(100), nb_steps(5)
{}

void MonteCarloPredictor::predict(const Eigen::VectorXd & input,
                                  std::shared_ptr<const Policy> policy,
                                  Problem::TransitionFunction transition_function,
                                  Problem::RewardFunction reward_function,
                                  Problem::ValueFunction value_function,
                                  Problem::TerminalFunction terminal_function,
                                  double discount,
                                  double * mean,
                                  double * var,
                                  std::default_random_engine * engine)
{
  if (!policy) {
    throw std::logic_error("MonteCarloPredictor cannot predict if it is given a null policy!");
  }
  std::vector<double> rewards(nb_predictions);
  // Preparing function:
  MonteCarloPredictor::RPTask prediction_task;
  prediction_task = getTask(input, policy, transition_function, reward_function, value_function,
                            terminal_function, discount, rewards);
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = rosban_random::getRandomEngines(std::min(nb_threads, nb_predictions), engine);
  // Now filling reward in parallel
  MultiCore::runParallelStochasticTask(prediction_task, nb_predictions, &engines);
  double internal_mean = regression_forests::Statistics::mean(rewards);
  if (mean != nullptr) *mean = internal_mean;
  if (var != nullptr) *var = regression_forests::Statistics::variance(rewards);
}

MonteCarloPredictor::RPTask
MonteCarloPredictor::getTask(const Eigen::VectorXd & input,
                             std::shared_ptr<const Policy> policy,
                             Problem::TransitionFunction transition_function,
                             Problem::RewardFunction reward_function,
                             Problem::ValueFunction value_function,
                             Problem::TerminalFunction terminal_function,
                             double discount,
                             std::vector<double> & rewards)
{
  return [this, input, policy, transition_function,
          reward_function, value_function, terminal_function,
          discount, &rewards]
    (int start_idx, int end_idx, std::default_random_engine * engine)
    {
      for (int prediction = start_idx; prediction < end_idx; prediction++)
      {
        double coeff = 1;
        double reward = 0;
        Eigen::VectorXd state = input;
        // Compute the reward over the next 'nb_steps'
        for (int i = 0; i < nb_steps; i++)
        {
          // Stop predicting steps if a terminal state has been reached
          if (terminal_function(state)) break;

          Eigen::VectorXd action = policy->getAction(state, engine);
          Eigen::VectorXd next_state = transition_function(state, action, engine);
          reward += coeff * reward_function(state, action, next_state);
          state = next_state;
          coeff *= discount;
        }
        // Use the value function to estimate long time reward
        if (!terminal_function(state)) reward += coeff * value_function(state);
        rewards[prediction] = reward;
      }
    };
}

std::string MonteCarloPredictor::class_name() const
{
  return "MonteCarloPredictor";
}

void MonteCarloPredictor::to_xml(std::ostream & out) const
{
  rosban_utils::xml_tools::write<int>("nb_predictions", nb_predictions, out);
  rosban_utils::xml_tools::write<int>("nb_steps"      , nb_steps      , out);
}

void MonteCarloPredictor::from_xml(TiXmlNode * node)
{
  rosban_utils::xml_tools::try_read<int>(node, "nb_predictions", nb_predictions);
  rosban_utils::xml_tools::try_read<int>(node, "nb_steps"      , nb_steps      );
}

}
