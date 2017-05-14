#include "rosban_csa_mdp/core/monte_carlo_policy.h"

#include "rosban_csa_mdp/core/policy_factory.h"
#include "rosban_csa_mdp/core/problem_factory.h"

#include "rosban_bbo/optimizer_factory.h"

namespace csa_mdp
{

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
        return this->averageReward(state, action_id, parameters, engine);
      };
    Eigen::VectorXd param_space =problem->getActionLimits(action_id);
    optimizer->setLimits(param_space);
    Eigen::VectorXd best_params = optimizer->train(eval_func, engine);
    double value = averageReward(state, action_id, best_params, engine);
    Eigen::VectorXd action(param_space.rows() + 1);
    action(0) = action_id;
    action.segment(1,param_space.rows()) = best_params;

    if (value > best_value) {
      best_value = value;
      best_action_id = action_id;
    }
  }
  return actions[best_action_id];
}

double MonteCarloPolicy::averageReward(const Eigen::VectorXd & initial_state,
                                       int action_id,
                                       const Eigen::VectorXd & params,
                                       std::default_random_engine * engine) const
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
                                      std::default_random_engine * engine) const
{
  double cumulated_reward = 0;
  Problem::Result result;
  // Perform the first step
  Eigen::VectorXd first_action(params.rows() + 1);
  first_action(0) = action_id;
  first_action.segment(1,params.rows()) = params;
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
}

}
