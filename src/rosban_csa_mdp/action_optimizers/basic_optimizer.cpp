#include "rosban_csa_mdp/action_optimizers/basic_optimizer.h"

#include "rosban_fa/function_approximator.h"
#include "rosban_fa/gp_trainer.h"
#include "rosban_fa/trainer_factory.h"

#include "rosban_regression_forests/tools/statistics.h"

#include "rosban_random/tools.h"

#include "rosban_utils/multi_core.h"

using rosban_fa::GPTrainer;
using rosban_fa::Trainer;
using rosban_fa::TrainerFactory;

using rosban_utils::MultiCore;

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
                                         Problem::TransitionFunction transition_function,
                                         Problem::RewardFunction reward_function,
                                         Problem::ValueFunction value_function,
                                         Problem::TerminalFunction terminal_function,
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
  AOTask task = getTask(input, actions, current_policy, transition_function, reward_function,
                        value_function, terminal_function, discount, results);
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
                                               Problem::TransitionFunction transition_function,
                                               Problem::RewardFunction reward_function,
                                               Problem::ValueFunction value_function,
                                               Problem::TerminalFunction terminal_function,
                                               double discount,
                                               Eigen::VectorXd & results) const
{
  return [this, input, actions, policy, transition_function,
          reward_function, value_function, terminal_function,
          discount, &results]
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
          Eigen::VectorXd state = transition_function(input, initial_action);
          total_reward += reward_function(input, initial_action, state);
          double coeff = discount;
          // 2. Using policy for a few steps
          for (int i = 0; i < this->nb_additional_steps; i++)
          {
            // Stop predicting steps if a terminal state has been reached
            if (terminal_function(state)) break;

            Eigen::VectorXd action = policy->getAction(state, engine);
            Eigen::VectorXd next_state = transition_function(state, action);
            total_reward += coeff * reward_function(state, action, next_state);
            state = next_state;
            coeff *= discount;
          }
          // 3. Using value at final state if provided
          if (value_function && !terminal_function(state)) {
            total_reward += coeff * value_function(state);
          }
        }
        results(action) = total_reward / nb_simulations;
      }
    };
}

std::string BasicOptimizer::class_name() const
{
  return "BasicOptimizer";
}

void BasicOptimizer::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("nb_additional_steps", nb_additional_steps, out);
  rosban_utils::xml_tools::write<int>("nb_simulations"     , nb_simulations     , out);
  rosban_utils::xml_tools::write<int>("nb_actions"         , nb_actions         , out);
  trainer->factoryWrite(trainer->class_name(), out);
}

void BasicOptimizer::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>(node, "nb_additional_steps", nb_additional_steps);
  rosban_utils::xml_tools::try_read<int>(node, "nb_simulations"     , nb_simulations     );
  rosban_utils::xml_tools::try_read<int>(node, "nb_actions"         , nb_actions         );
  TrainerFactory().tryRead(node, "trainer", trainer);
}

}
