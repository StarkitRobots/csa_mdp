#include "rosban_csa_mdp/action_optimizers/basic_optimizer.h"

#include "rosban_fa/function_approximator.h"

#include "rosban_random/tools.h"


namespace csa_mdp
{

BasicOptimizer::BasicOptimizer()
  : nb_additional_steps(4),
    nb_simulations(100),
    nb_actions(25)
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
    // Compute several simulations
    for (int sim = 0; sim < nb_simulations; sim++) {
      // 1. Using chosen action
      Eigen::VectorXd state = model->getSuccessor(input, initial_action);
      double reward = reward_function(input, initial_action, state);
      double coeff = discount;
      // 2. Using current_policy for a few steps
      for (int i = 0; i < nb_additional_steps; i++)
      {
        Eigen::VectorXd action = current_policy->getAction(state, &engine);
        Eigen::VectorXd next_state = model->getSuccessor(state, action);
        reward += coeff * reward_function(state, action, next_state);
        state = next_state;
        coeff *= discount;
      }
      // 3. Using value at final state if provided
      if (value_function) reward += coeff * value_function(state);
      results(action) += reward;
    }
  }
  // Averaging the rewards is useless
  // Train a function approximator
  std::unique_ptr<rosban_fa::FunctionApproximator> approximator;
  approximator = gp_trainer.train(actions, results, model->getActionLimits());
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
  gp_trainer.write("gp_trainer", out);
}

void BasicOptimizer::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>(node, "nb_additional_steps", nb_additional_steps);
  rosban_utils::xml_tools::try_read<int>(node, "nb_simulations"     , nb_simulations     );
  rosban_utils::xml_tools::try_read<int>(node, "nb_actions"         , nb_actions         );
  gp_trainer.tryRead(node, "gp_trainer");
}

}
