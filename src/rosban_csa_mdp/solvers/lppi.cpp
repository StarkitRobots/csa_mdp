#include "rosban_csa_mdp/solvers/lppi.h"

#include "rosban_fa/constant_approximator.h"
#include "rosban_fa/function_approximator.h"
#include "rosban_fa/trainer_factory.h"

#include "rosban_random/tools.h"

#include "rhoban_utils/threading/multi_core.h"
#include "rhoban_utils/timing/time_stamp.h"

#include <mutex>

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

void LPPI::performRollout(Eigen::MatrixXd * states,
                          Eigen::MatrixXd * actions,
                          Eigen::VectorXd * values,
                          std::default_random_engine * engine)
{
  int state_dims = problem->stateDims();
  int action_dims = problem->actionDims(0);
  // First, run the rollout storing visited states
  std::vector<Eigen::VectorXd> rollout_states, rollout_actions;
  std::vector<double> rollout_rewards;
  Eigen::VectorXd state = problem->getStartingState(engine);
  bool end_with_terminal = true;
  for (int step = 0; step < max_rollout_length; step++) {
    // Local optimization of the action
    Eigen::VectorXd action = planner.planNextAction(*problem, state, *value, engine);
    // Applying action, storing results and updating current state
    Problem::Result res = problem->getSuccessor(state, action, engine);
    rollout_states.push_back(state);
    rollout_actions.push_back(action);
    rollout_rewards.push_back(res.reward);
    // Stop if we obtained a terminal status, otherwise, update current state
    if (res.terminal) {
      end_with_terminal = true;
      break;
    } else {
      state = res.successor;
    }
  }
  // Now, fill states, actions and values by going back
  int rollout_length = rollout_states.size();
  int last_idx_used = rollout_length -1;
  double value = 0;
  // If end of rollout was the result of an horizon end, ensure that
  // states, actions and values returned 
  if (!end_with_terminal) {
    last_idx_used = rollout_length - min_rollout_length;
    for (int idx = rollout_length-1; idx > last_idx_used; idx--) {
      value = value * discount + rollout_rewards[idx];
    }
  }
  std::cout << "-----" << std::endl
            << "rollout_length : " << rollout_length << std::endl
            << "last_idx_used : " << last_idx_used << std::endl
            << "value : " << value << std::endl;
  // Initalize and fill results
  int rollout_entries = last_idx_used + 1;
  (*states)  = Eigen::MatrixXd(state_dims , rollout_entries);
  (*actions) = Eigen::MatrixXd(1+action_dims, rollout_entries);
  (*values)  = Eigen::VectorXd(rollout_entries);
  for (int idx = last_idx_used; idx >= 0; idx--) {
    value = value * discount + rollout_rewards[idx];
    states->col(idx) = rollout_states[idx];
    actions->col(idx) = rollout_actions[idx];
    (*values)(idx) = value;
  }
}

void LPPI::performRollouts(Eigen::MatrixXd * states,
                           Eigen::MatrixXd * actions,
                           Eigen::VectorXd * values,
                           std::default_random_engine * engine)
{
  int state_dims = problem->stateDims();
  int action_dims = problem->actionDims(0);
  int entry_count = 0;
  (*states)  = Eigen::MatrixXd(state_dims , nb_entries);
  (*actions) = Eigen::MatrixXd(1+action_dims, nb_entries);
  (*values)  = Eigen::VectorXd(nb_entries);
  std::mutex mutex;// Ensures only one thread modifies common properties at the same time
  // TODO: add another StochasticTask which does not depend on start_idx and end_idx eventually
  rhoban_utils::MultiCore::StochasticTask thread_task =
    [this, &mutex, &state_dims, &action_dims, states, actions, values, &entry_count]
    (int start_idx, int end_idx, std::default_random_engine * engine)
    {
      (void) start_idx;(void) end_idx;
      while (entry_count < this->nb_entries) {
        Eigen::MatrixXd rollout_states, rollout_actions;
        Eigen::VectorXd rollout_values;
        this->performRollout(&rollout_states, &rollout_actions, &rollout_values, engine);
        int nb_new_entries = rollout_states.cols();
        // Updating content
        mutex.lock();
        if (entry_count >= this->nb_entries) {
          mutex.unlock();
          return;
        }
        // If there is too much new entries, remove some to have exactly the
        // requested number
        if (entry_count + nb_new_entries > this->nb_entries) {
          nb_new_entries = this->nb_entries - entry_count;
          rollout_states  = rollout_states.block (0, 0, state_dims     , nb_new_entries);
          rollout_actions = rollout_actions.block(0, 0, 1 + action_dims, nb_new_entries);
          rollout_values  = rollout_values.segment(0, nb_new_entries);
        }
        states ->block(0,entry_count,state_dims   , nb_new_entries) = rollout_states ;
        actions->block(0,entry_count,1+action_dims, nb_new_entries) = rollout_actions;
        values->segment(entry_count, nb_new_entries) = rollout_values;
        entry_count += nb_new_entries;
        std::cout << "Entry count: " << entry_count << std::endl;
        mutex.unlock();
      }
    };
  std::vector<std::default_random_engine> engines;
  int local_nb_threads = std::min(nb_threads, nb_entries);
  engines = rosban_random::getRandomEngines(local_nb_threads, engine);
  rhoban_utils::MultiCore::runParallelStochasticTask(thread_task, local_nb_threads, &engines);
}
void LPPI::init(std::default_random_engine * engine) {
  if (problem->getNbActions() != 1) {
    throw std::runtime_error("LPPI::performRollouts: no support for hybrid action spaces");
  }
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
