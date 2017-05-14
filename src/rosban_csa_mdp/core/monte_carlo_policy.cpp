#include "rosban_csa_mdp/core/monte_carlo_policy.h"

#include "rosban_csa_mdp/core/problem.h"

namespace csa_mdp
{

void MonteCarloPolicy::init()
{
  std::random_device rd;
  internal_engine.seed(rd);
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
        return averageReward(state, action_id, parameters, engine);
      };
    Eigen::VectorXd param_space =problem->getActionSpace(action_id);
    optimizer->setLimits(param_space);
    Eigen::VectorXd best_params = optimizer->train(action_id);
    double value = averageReward(state, action_id, best_parameters, engine);
    Eigen::VectorXd action(param_space.rows() + 1);
    action(0) = action_id;
    action.segment(1,param_space.rows());

    if (value > lowest) {
      lowest = value;
      best_action_id = action_id;
    }
  }
  return actions[best_action_id];
}

double MonteCarloPolicy::averageReward(const Eigen::VectorXd & initial_state,
                                       int action_id,
                                       const Eigen::VectorXd & params,
                                       std::default_random_engine * engine)
{
  double total_reward = 0;
  for (int rollout = 0; rollout < nb_rollouts; rollout++) {
    total_reward += sampleReward(initial_state, action_id, params, engine);
  }
  return total_reward / nb_rollouts;
}


double MonteCarloPolicy::sampleReward(const Eigen::VectorXd & initial_state,
                                      int action_id,
                                      const Eigen::VectorXd & params,
                                      std::default_random_engine * engine)
{
  double cumulated_reward = 0;
  Problem::Result result;
  // Perform the first step
  Eigen::VectorXd first_action(params.rows() + 1);
  first_action(0) = action_id;
  first_action.segment(1,params.rows()) = params.rows();
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
    cumulated_reward += result;
  }
  return cumulated_reward;
}


}
