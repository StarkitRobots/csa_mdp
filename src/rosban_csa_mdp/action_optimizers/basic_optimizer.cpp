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
{
  engine = rosban_random::getRandomEngine();
}

Eigen::VectorXd BasicOptimizer::optimize(const Eigen::VectorXd & input,
                                         std::shared_ptr<const Policy> current_policy,
                                         std::shared_ptr<Problem> model,
                                         RewardFunction reward_function,
                                         ValueFunction value_function,
                                         double discount)
{
  Eigen::MatrixXd actions = rosban_random::getUniformSamplesMatrix(model->getActionLimits(),
                                                                   nb_actions,
                                                                   &engine);
  Eigen::VectorXd results = Eigen::VectorXd::Zero(nb_actions);
  for (int action = 0; action < nb_actions; action++)
  {
    Eigen::VectorXd initial_action = actions.col(action);
    // Prepare simulations data
    std::vector<double> rewards(nb_simulations);
    std::vector<std::default_random_engine> engines;
    engines = rosban_random::getRandomEngines(nb_threads, &engine);
    // Preparing function:
    auto simulator =
      [this, input, initial_action, current_policy, model, reward_function, value_function,
       discount, &rewards]
      (int start_idx, int end_idx, std::default_random_engine * engine)
    {
      // Compute several simulations
      for (int sim = start_idx; sim < end_idx; sim++) {
        // 1. Using chosen action
        Eigen::VectorXd state = model->getSuccessor(input, initial_action);
        double reward = reward_function(input, initial_action, state);
        double coeff = discount;
        // 2. Using current_policy for a few steps
        for (int i = 0; i < this->nb_additional_steps; i++)
        {
          Eigen::VectorXd action = current_policy->getAction(state, engine);
          Eigen::VectorXd next_state = model->getSuccessor(state, action);
          reward += coeff * reward_function(state, action, next_state);
          state = next_state;
          coeff *= discount;
        }
        // 3. Using value at final state if provided
        if (value_function) reward += coeff * value_function(state);
        rewards[sim] += reward;
      }
    };// End of simulator
    // Now filling reward in parallel
    MultiCore::runParallelStochasticTask(simulator, nb_simulations, &engines);
    results(action) = regression_forests::Statistics::mean(rewards);
  }
  // Averaging the rewards is useless
  // Train a function approximator
  std::unique_ptr<rosban_fa::FunctionApproximator> approximator;
  approximator = trainer->train(actions, results, model->getActionLimits());
  Eigen::VectorXd best_guess;
  double best_output;
  approximator->getMaximum(model->getActionLimits(),
                           best_guess,
                           best_output);
  return best_guess;
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
