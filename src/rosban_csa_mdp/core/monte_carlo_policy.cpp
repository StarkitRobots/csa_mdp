#include "rosban_csa_mdp/core/monte_carlo_policy.h"

#include "rosban_csa_mdp/core/policy_factory.h"
#include "rosban_csa_mdp/core/problem_factory.h"

#include "rosban_bbo/optimizer_factory.h"

#include "rosban_random/tools.h"
#include "rhoban_utils/threading/multi_core.h"

namespace csa_mdp
{

MonteCarloPolicy::MonteCarloPolicy()
  : nb_rollouts(1), max_evals(1000), validation_rollouts(10),
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
  if (debug_level >= 2) {
    std::cout << "Optimization for state: " << state.transpose() << std::endl;
  }

  // Computing best actions and associatied values for each action
  std::vector<double> action_rewards;
  std::vector<Eigen::VectorXd> actions;
  int best_action_id = -1;
  double best_value = std::numeric_limits<double>::lowest();
  optimizer->setMaxCalls(max_evals / problem->getNbActions());
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
  // Simple version for mono-threading or mono rollout
  if (nb_threads == 1 || rollouts == 1) {
    double total_reward = 0;
    for (int rollout = 0; rollout < rollouts; rollout++) {
      total_reward += sampleReward(initial_state, first_action, engine);
    }
    return total_reward / rollouts;
  }
  // Preparing random_engines + rewards storing
  std::vector<std::default_random_engine> engines;
  engines = rosban_random::getRandomEngines(std::min(nb_threads, rollouts), engine);
  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(rollouts);
  // The task which has to be performed :
  rhoban_utils::MultiCore::StochasticTask task =
    [this, &initial_state, &first_action, &rewards]
    (int start_idx, int end_idx, std::default_random_engine * engine)
    {
      for (int idx = start_idx; idx < end_idx; idx++) {
        rewards(idx) = sampleReward(initial_state, first_action, engine);
      }
    };
  // Running computation
  rhoban_utils::MultiCore::runParallelStochasticTask(task, rollouts, &engines);

  return rewards.mean();
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

std::string MonteCarloPolicy::getClassName() const
{
  return "MonteCarloPolicy";
}

Json::Value MonteCarloPolicy::toJson() const
{
  Json::Value v = Policy::toJson();
  throw std::logic_error("MonteCarloPolicy::toJson: not implemented");
}

void MonteCarloPolicy::fromJson(const Json::Value & v, const std::string & dir_name)
{
  Policy::fromJson(v,dir_name);
  // Read problem directly from node or from another file
  std::string problem_path;
  rhoban_utils::tryRead<std::string>(v, "problem_path", &problem_path);
  if (problem_path != "") {
    problem = ProblemFactory().buildFromJsonFile(problem_path);
  } else {
    problem = ProblemFactory().read(v, "problem", dir_name);
  }
  // Mandatory elements
  default_policy = PolicyFactory().read(v, "default_policy", dir_name);
  optimizer = rosban_bbo::OptimizerFactory().read(v, "optimizer", dir_name);
  nb_rollouts         = rhoban_utils::read<int>(v, "nb_rollouts");
  validation_rollouts = rhoban_utils::read<int>(v, "validation_rollouts");
  simulation_depth    = rhoban_utils::read<int>(v, "simulation_depth");
  // Optional elements
  rhoban_utils::tryRead(v, "max_evals"  , &max_evals  );
  rhoban_utils::tryRead(v, "debug_level", &debug_level);


  if (!default_policy || !optimizer || !problem) {
    throw std::runtime_error("MonteCarloPolicy::fromJson: incomplete initialization");
  }

  default_policy->setActionLimits(problem->getActionsLimits());
}

}
