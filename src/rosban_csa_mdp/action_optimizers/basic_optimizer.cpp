#include "rosban_csa_mdp/action_optimizers/basic_optimizer.h"

#include "rosban_fa/function_approximator.h"
#include "rosban_fa/gp_trainer.h"
#include "rosban_fa/trainer_factory.h"

#include "rosban_regression_forests/tools/statistics.h"

#include "rosban_random/tools.h"

#include "rhoban_utils/threading/multi_core.h"

using rosban_fa::GPTrainer;
using rosban_fa::Trainer;
using rosban_fa::TrainerFactory;

using rhoban_utils::MultiCore;

namespace csa_mdp
{

BasicOptimizer::BasicOptimizer()
  : nb_additional_steps(4),
    nb_simulations(100),
    nb_actions(15),
    trainer(new GPTrainer)
{}

Eigen::VectorXd BasicOptimizer::optimize(const Eigen::VectorXd & input,
                                         const Eigen::MatrixXd & action_limits,
                                         std::shared_ptr<const Policy> current_policy,
                                         Problem::ResultFunction result_function,
                                         Problem::ValueFunction value_function,
                                         double discount,
                                         std::default_random_engine * engine) const
{
  bool clean_engine = false;
  if (engine == nullptr) {
    engine = rosban_random::newRandomEngine();
    clean_engine = true;
  }

  // actionDim by nb_actions
  Eigen::MatrixXd actions = rosban_random::getUniformSamplesMatrix(action_limits,
                                                                   nb_actions,
                                                                   engine);
  Eigen::VectorXd results = Eigen::VectorXd::Zero(nb_actions);
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = rosban_random::getRandomEngines(std::min(nb_threads, nb_actions), engine);
  // Preparing function:
  AOTask task = getTask(input, actions, current_policy, result_function,
                        value_function, discount, results);
  // Now filling reward in parallel
  MultiCore::runParallelStochasticTask(task, nb_actions, &engines);
  // Train a function approximator
  std::unique_ptr<rosban_fa::FunctionApproximator> approximator;
  approximator = trainer->train(actions, results, action_limits);
  Eigen::VectorXd best_guess;
  double best_output;
  approximator->getMaximum(action_limits, best_guess, best_output);
  if (clean_engine) delete(engine);


  // Debug:
  //std::ostringstream oss;
  //oss << "-----------------------" << std::endl
  //    << "Optimizing action in state: " << input.transpose() << std::endl;
  //for (int i = 0; i < nb_actions; i++) {
  //  oss << "\tAction '" << actions.col(i).transpose() << "' -> " << results(i) << std::endl;
  //}
  //oss << "Best action: " << best_guess.transpose() << std::endl
  //    << "-----------------------" << std::endl;
  //std::cout << oss.str();
  return best_guess;
}

BasicOptimizer::AOTask BasicOptimizer::getTask(const Eigen::VectorXd & input,
                                               const Eigen::MatrixXd & actions,
                                               std::shared_ptr<const Policy> policy,
                                               Problem::ResultFunction result_function,
                                               Problem::ValueFunction value_function,
                                               double discount,
                                               Eigen::VectorXd & results) const
{
  return [this, input, actions, policy, result_function,
          value_function, discount, &results]
    (int start_idx, int end_idx, std::default_random_engine * engine)
    {
      for (int action = start_idx; action < end_idx; action++)
      {
        Eigen::VectorXd initial_action = actions.col(action);
        // Prepare simulations data
        double total_reward = 0;
        // Compute several simulations
        for (int sim = 0; sim < nb_simulations; sim++) {
          // 1. Using chosen action
          Problem::Result result = result_function(input, initial_action, engine);
          total_reward += result.reward;
          double coeff = discount;
          // 2. Using policy for a few steps
          for (int i = 0; i < this->nb_additional_steps; i++)
          {
            // Stop predicting steps if a terminal state has been reached
            if (result.terminal) break;

            Eigen::VectorXd action = policy->getAction(result.successor, engine);
            result = result_function(result.successor, action, engine);
            total_reward += coeff * result.reward;
            coeff *= discount;
          }
          // 3. Using value at final state if provided
          if (value_function && !result.terminal) {
            total_reward += coeff * value_function(result.successor);
          }
        }
        results(action) = total_reward / nb_simulations;
      }
    };
}

std::string BasicOptimizer::getClassName() const
{
  return "BasicOptimizer";
}

Json::Value BasicOptimizer::toJson() const
{
  Json::Value v;
  v["nb_additional_steps"] = nb_additional_steps;
  v["nb_simulations"     ] = nb_simulations     ;
  v["nb_actions"         ] = nb_actions         ;
  v["trainer"] = trainer->toFactoryJson();
  return v;
}

void BasicOptimizer::fromJson(const Json::Value & v, const std::string & dir_name)
{
  rhoban_utils::tryRead(v, "nb_additional_steps", &nb_additional_steps);
  rhoban_utils::tryRead(v, "nb_simulations"     , &nb_simulations     );
  rhoban_utils::tryRead(v, "nb_actions"         , &nb_actions         );
  TrainerFactory().tryRead(v, "trainer", dir_name, &trainer);
}

}
