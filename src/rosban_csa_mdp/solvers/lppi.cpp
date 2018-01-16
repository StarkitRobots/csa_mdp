#include "rosban_csa_mdp/solvers/lppi.h"

#include "rosban_fa/constant_approximator.h"
#include "rosban_fa/function_approximator.h"
#include "rosban_fa/trainer_factory.h"

#include "rhoban_utils/timing/time_stamp.h"

using namespace rosban_fa;
using rhoban_utils::TimeStamp;

namespace csa_mdp
{

LPPI::LPPI() : min_rollout_length(-1), max_rollout_length(-1), nb_entries(-1)
{
}

LPPI::~LPPI()
{
}

void LPPI::performRollouts(Eigen::MatrixXd * states,
                           Eigen::MatrixXd * actions,
                           Eigen::VectorXd * values,
                           std::default_random_engine * engine)
{
  if (problem->getNbActions() != 1) {
    throw std::runtime_error("LPPI::performRollouts: no support for hybrid action spaces");
  }
  int state_dims = problem->stateDims();
  int action_dims = problem->actionDims(0);
  int entry_idx = 0;
  (*states)  = Eigen::MatrixXd(state_dims , nb_entries);
  (*actions) = Eigen::MatrixXd(1+action_dims, nb_entries);
  (*values)  = Eigen::VectorXd(nb_entries);
  while(entry_idx < nb_entries) {
    // First, run the rollout storing visited states
    std::vector<Eigen::VectorXd> rollout_states, rollout_actions;
    std::vector<double> rollout_rewards;
    Eigen::VectorXd state = problem->getStartingState(engine);
    bool trial_interrupted = true;
    for (int step = 0; step < max_rollout_length; step++) {
      // Local optimization of the action
      Eigen::VectorXd action;
      action = planner.planNextAction(*problem, state, *value, engine);
      Eigen::VectorXd prefixed_action(1+action_dims);
      prefixed_action(0) = 0;
      prefixed_action.segment(1,action_dims);
      // Applying action, storing results and updating current state
      Problem::Result res = problem->getSuccessor(state, prefixed_action, engine);
      rollout_states.push_back(state);
      rollout_actions.push_back(prefixed_action);
      rollout_rewards.push_back(res.reward);
      // Stop if we obtained a terminal status, otherwise, update current state
      if (res.terminal) {
        trial_interrupted = true;
        std::cout << "Trial interrupted in state : " << res.successor.transpose() << std::endl;
        break;
      } else {
        state = res.successor;
      }
    }
    // Now, fill states, actions and values by going back
    double value = 0;
    int rollout_length = rollout_states.size();
    for (int idx = rollout_length-1; idx >= 0; idx--) {
      value = value * discount + rollout_rewards[idx];
      // If entry is allowed, push it in results
      if (trial_interrupted || (rollout_length - idx) > min_rollout_length) {
        states->col(entry_idx) = rollout_states[idx];
        actions->col(entry_idx) = rollout_actions[idx];
        (*values)(entry_idx) = value;
        entry_idx++;
        if (entry_idx >= nb_entries) {
          break;
        }
      }
    }
    std::cout << "Entry_idx: " << entry_idx << std::endl;
  }
}
void LPPI::init(std::default_random_engine * engine) {
  (void)engine;
  if (!value) {
    Eigen::VectorXd default_value(1);
    default_value(0) = 0;
    value = std::unique_ptr<FunctionApproximator>(new ConstantApproximator(default_value));
  }
}

void LPPI::update(std::default_random_engine * engine) {
  // Acquiring entries by performing rollouts with online planner
  Eigen::MatrixXd states, actions;
  Eigen::VectorXd values;
  TimeStamp start = TimeStamp::now();
  performRollouts(&states, &actions, &values, engine);
  TimeStamp mid1 = TimeStamp::now();
  // Updating both policy and value based on actions
  Eigen::MatrixXd state_limits = problem->getStateLimits();
  value = value_trainer->train(states, values, state_limits);
  policy = policy_trainer->train(states, actions.transpose(), state_limits);
  TimeStamp mid2 = TimeStamp::now();
  std::unique_ptr<Policy> new_policy = buildPolicy(*policy);
  double new_reward = evaluatePolicy(*new_policy, engine);
  TimeStamp end = TimeStamp::now();
  std::cout << "New reward: " << new_reward << std::endl;
  policy->save("policy_tree.bin");
  writeTime("performRollouts"  , diffSec(start , mid1 ));
  writeTime("updateValues"     , diffSec(mid1  , mid2 ));
  writeTime("evalPolicy"       , diffSec(mid2  , end  ));
  writeScore(new_reward);
}

void LPPI::setNbThreads(int nb_threads) {
  BlackBoxLearner::setNbThreads(nb_threads);
  if (value_trainer) {
    value_trainer->setNbThreads(nb_threads);
  }
  if (policy_trainer) {
    policy_trainer->setNbThreads(nb_threads);
  }
  //TODO: use open_loop_planner.setNbThreads
}

std::string LPPI::getClassName() const {
  return "LPPI";
}

Json::Value LPPI::toJson() const {
  Json::Value v = BlackBoxLearner::toJson();
  throw std::runtime_error("LPPI::toJson: Not implemented");
}

void LPPI::fromJson(const Json::Value & v, const std::string & dir_name) {
  BlackBoxLearner::fromJson(v, dir_name);
  planner.read(v, "planner", dir_name);
  TrainerFactory().tryRead(v, "value_trainer" , dir_name, &value_trainer );
  TrainerFactory().tryRead(v, "policy_trainer", dir_name, &policy_trainer);
  rhoban_utils::tryRead(v, "min_rollout_length", &min_rollout_length);
  rhoban_utils::tryRead(v, "max_rollout_length", &max_rollout_length);
  rhoban_utils::tryRead(v, "nb_entries"        , &nb_entries        );
}

}
