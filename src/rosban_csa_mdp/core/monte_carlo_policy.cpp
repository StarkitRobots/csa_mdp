#include "rosban_csa_mdp/core/monte_carlo_policy.h"

#include "rosban_csa_mdp/core/policy_factory.h"
#include "rosban_csa_mdp/core/problem_factory.h"

#include "rosban_bbo/optimizer_factory.h"

namespace csa_mdp
{

MonteCarloPolicy::MonteCarloPolicy()
  : nb_rollouts(1), validation_rollouts(10),
    simulation_depth(1), debug_level(0)
{
}

void MonteCarloPolicy::init()
{
  std::random_device rd;
  internal_engine.seed(rd());
}

Eigen::VectorXd MonteCarloPolicy::getRawAction(const Eigen::VectorXd &state)
{
  return getRawAction(state, &internal_engine);
}

Eigen::VectorXd MonteCarloPolicy::getRawAction(const Eigen::VectorXd &state,
                                               std::default_random_engine * engine) const
{
  // Computing best actions and associatied values for each action
  std::vector<double> action_rewards;
  std::vector<Eigen::VectorXd> actions;
  int best_action_id = -1;
  double best_value = std::numeric_limits<double>::lowest();
  for(int action_id = 0; action_id < problem->getNbActions(); action_id++) {
    rosban_bbo::Optimizer::RewardFunc eval_func =
      [&](const Eigen::VectorXd & parameters,
          std::default_random_engine * engine)
      {
        Eigen::VectorXd action(parameters.rows() + 1);
        action(0) = action_id;
        action.segment(1,parameters.rows()) = parameters;
        return this->averageReward(state, action, nb_rollouts, engine);
      };
    Eigen::MatrixXd param_space = problem->getActionLimits(action_id);
    optimizer->setLimits(param_space);
    Eigen::VectorXd params = optimizer->train(eval_func, engine);
    Eigen::VectorXd action(params.rows()+1);
    action(0) = action_id;
    action.segment(1,params.rows()) = params;
    double value = averageReward(state, action, validation_rollouts, engine);

    actions.push_back(action);
    action_rewards.push_back(value);

    if (debug_level >= 2) {
      std::cout << "Choice: " << action_id << ": " << action.transpose()
                << " -> " << value << std::endl;
    }

    if (value > best_value) {
      best_value = value;
      best_action_id = action_id;
    }
  }

  Eigen::VectorXd original_action = default_policy->getAction(state, engine);
  double original_value = averageReward(state, original_action,
                                        validation_rollouts,
                                        engine);
  if (debug_level >= 2) {
    std::cout << "Default: " << original_action.transpose()
              << " -> " << original_value << std::endl;
  }
  if (debug_level >= 1) {
    std::cout << "MCOptimize gain: " << (best_value - original_value) << std::endl;
  }
  if (original_value > best_value) {
    return original_action;
  }

  return actions[best_action_id];
}

double MonteCarloPolicy::averageReward(const Eigen::VectorXd & initial_state,
                                       const Eigen::VectorXd & first_action,
                                       int rollouts,
                                       std::default_random_engine * engine) const
{
  double total_reward = 0;
  for (int rollout = 0; rollout < rollouts; rollout++) {
    total_reward += sampleReward(initial_state, first_action, engine);
  }
  return total_reward / rollouts;
}


double MonteCarloPolicy::sampleReward(const Eigen::VectorXd & initial_state,
                                      const Eigen::VectorXd & first_action,
                                      std::default_random_engine * engine) const
{
  double cumulated_reward = 0;
  Problem::Result result;
  // Perform the first step
  result = problem->getSuccessor(initial_state,
                                 first_action,
                                 engine);
  cumulated_reward = result.reward;
  // Perform remaining steps
  for (int step = 1; step < simulation_depth; step++) {
    // End evaluation as soon as a terminal state has been reached
    if (result.terminal) break;
    // Compute a single step
    Eigen::VectorXd action = default_policy->getAction(result.successor, engine);
    result = problem->getSuccessor(result.successor, action, engine);
    cumulated_reward += result.reward;
  }
  return cumulated_reward;
}

std::string MonteCarloPolicy::class_name() const
{
  return "monte_carlo_policy";
}

void MonteCarloPolicy::to_xml(std::ostream & out) const
{
  (void)out;
  throw std::logic_error("MonteCarloPolicy::to_xml: not implemented");
}

void MonteCarloPolicy::from_xml(TiXmlNode * node)
{
  problem = ProblemFactory().read(node, "problem");
  default_policy = PolicyFactory().read(node, "default_policy");
  optimizer = rosban_bbo::OptimizerFactory().read(node, "optimizer");
  nb_rollouts = rosban_utils::xml_tools::read<int>(node, "nb_rollouts");
  validation_rollouts = rosban_utils::xml_tools::read<int>(node, "validation_rollouts");
  simulation_depth = rosban_utils::xml_tools::read<int>(node, "simulation_depth");
  debug_level = rosban_utils::xml_tools::read<int>(node, "debug_level");

  if (!default_policy || !optimizer || !problem) {
    throw std::runtime_error("MonteCarloPolicy::from_xml: incomplete initialization");
  }

  default_policy->setActionLimits(problem->getActionsLimits());
}

}
