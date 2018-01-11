#include "rosban_csa_mdp/solvers/fpf.h"

#include "rosban_gp/gradient_ascent/randomized_rprop.h"

#include "rosban_random/tools.h"

#include "rhoban_utils/timing/benchmark.h"
#include "rhoban_utils/threading/multi_core.h"

#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

using rhoban_utils::Benchmark;
using rhoban_utils::MultiCore;
using rhoban_utils::TimeStamp;

using regression_forests::Approximation;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;

namespace csa_mdp
{

FPF::Config::Config()
{
  nb_threads = 1;
  x_dim = 0;
  u_dim = 0;
  horizon = 1;
  discount = 0.99;
  max_action_tiles = 0;
  policy_samples = 0;
  q_training_set_time = 0;
  q_extra_trees_time  = 0;
  p_training_set_time = 0;
  p_extra_trees_time  = 0;
  auto_parameters = true;
  gp_values = false;
  gp_policies = false;
}

const Eigen::MatrixXd & FPF::Config::getStateLimits() const
{
  return x_limits;
}

const Eigen::MatrixXd & FPF::Config::getActionLimits() const
{
  return u_limits;
}
Eigen::MatrixXd FPF::Config::getInputLimits() const
{
  // Init Knownness Forest
  int s_dim = getStateLimits().rows();
  int a_dim = getActionLimits().rows();
  Eigen::MatrixXd limits(s_dim + a_dim, 2);
  limits.block(    0, 0, s_dim, 2) = getStateLimits();
  limits.block(s_dim, 0, a_dim, 2) = getActionLimits();
  return limits;
}

void FPF::Config::setStateLimits(const Eigen::MatrixXd &new_limits)
{
  x_limits = new_limits;
  x_dim = x_limits.rows();
}

void FPF::Config::setActionLimits(const Eigen::MatrixXd &new_limits)
{
  u_limits = new_limits;
  u_dim = u_limits.rows();
}

std::string FPF::Config::getClassName() const
{
  return "FPFConfig";
}

Json::Value FPF::Config::toJson() const
{
  Json::Value v;
  // Writing limits
  v["x_limits"] = rhoban_utils::matrix2Json(x_limits);
  v["u_limits"] = rhoban_utils::matrix2Json(u_limits);
  // Writing properties
  v["horizon"         ] = (int)horizon         ;
  v["nb_threads"      ] = nb_threads           ;
  v["discount"        ] = discount             ;
  v["policy_samples"  ] = policy_samples       ;
  v["max_action_tiles"] = (int)max_action_tiles;
  v["auto_parameters" ] = auto_parameters      ;
  v["gp_values"       ] = gp_values            ;
  v["gp_policies"     ] = gp_policies          ;
  // If parameters are not auto, writing parameters for forests training
  if (!auto_parameters)
  {
    v["q_value_conf"] = q_value_conf.toJson();
    v["policy_conf"] = policy_conf.toJson();
  }
  if (gp_values) {
    v["find_max_rprop_conf"] = find_max_rprop_conf.toJson();
  }
  if (gp_values || gp_policies) {
    v["hyper_rprop_conf"] = hyper_rprop_conf.toJson();
  }
  return v;
}

void FPF::Config::fromJson(const Json::Value & v, const std::string & dir_name)
{
  // Reading limits of the problem
  x_limits = rhoban_utils::read<Eigen::MatrixXd>(v,"x_limits");
  u_limits = rhoban_utils::read<Eigen::MatrixXd>(v,"u_limits");
  // Reading mandatory properties
  horizon          = rhoban_utils::read<int>   (v, "horizon"         );
  discount         = rhoban_utils::read<double>(v, "discount"        );
  max_action_tiles = rhoban_utils::read<int>   (v, "max_action_tiles");
  // Reading optional properties
  rhoban_utils::tryRead(v, "nb_threads"      , &nb_threads      );
  rhoban_utils::tryRead(v, "policy_samples"  , &policy_samples  );
  rhoban_utils::tryRead(v, "auto_parameters" , &auto_parameters );
  if (!auto_parameters)
  {
    q_value_conf.read(v, "q_value_conf", dir_name);
    policy_conf.read(v, "policy_conf", dir_name);
  }
  rhoban_utils::tryRead(v, "gp_values"   , &gp_values);
  rhoban_utils::tryRead(v, "gp_policies" , &gp_policies);
  if (gp_values) {
    find_max_rprop_conf.tryRead(v, "find_max_rprop_conf", dir_name);
  }
  if (gp_values || gp_policies) {
    hyper_rprop_conf.tryRead(v, "hyper_rprop_conf", dir_name);
  }
}

FPF::FPF()
{
}

const regression_forests::Forest& FPF::getValueForest()
{
  return *q_value;
}

const regression_forests::Forest& FPF::getPolicyForest(int action_index)
{
  if (action_index > (int)policies.size())
    throw std::out_of_range("action_index greater than number of policies");
  return *(policies[action_index]);
}

std::unique_ptr<regression_forests::Forest> FPF::stealPolicyForest(int action_index)
{
  return std::unique_ptr<regression_forests::Forest>(policies[action_index].release());
}

void FPF::updateQValue(const std::vector<Sample>& samples,
                       std::function<bool(const Eigen::VectorXd&)> isTerminal,
                       Config &conf,
                       bool last_step)
{
  regression_forests::ExtraTrees q_learner;
  if (conf.auto_parameters)
  {
    // Using piecewise linear approximation is only allowed during the last step
    Approximation::ID appr_type = last_step ? Approximation::ID::PWL : Approximation::ID::PWC;
    // TODO implement alternative update with PWL approximations to allow their use
    appr_type = Approximation::ID::PWC;
    // Replacing by GP if required
    if (conf.gp_values) appr_type = Approximation::ID::GP;
    // Generating the configuration automatically
    q_learner.conf = ExtraTrees::Config::generateAuto(conf.getInputLimits(),
                                                      samples.size(),
                                                      appr_type);
    q_learner.conf.nb_threads = conf.nb_threads;
    // If using GP, use the custom parameters for hyperparameters tuning
    q_learner.conf.gp_conf = conf.hyper_rprop_conf;
  }
  else
  {
    q_learner.conf = conf.q_value_conf;
  }
  // ts is computed using last q_value
  Benchmark::open("Getting TrainingSet");
  TrainingSet ts = getTrainingSet(samples, isTerminal, conf);
  conf.q_training_set_time += Benchmark::close();
  Benchmark::open("q_learner.solve()");
  q_value = q_learner.solve(ts, conf.getInputLimits());
  conf.q_extra_trees_time += Benchmark::close();
}

void FPF::solve(const std::vector<Sample>& samples,
                std::function<bool(const Eigen::VectorXd&)> isTerminal,
                Config &conf)
{
  // Resetting properties
  //q_value.release();//Experimental
  conf.q_training_set_time = 0;
  conf.q_extra_trees_time  = 0;
  conf.p_training_set_time = 0;
  conf.p_extra_trees_time  = 0;
  // Updating q-value
  Benchmark::open("Updating Q-Value");
  for (size_t h = 1; h <= conf.horizon; h++) {
    bool last_step = (h == conf.horizon);
    updateQValue(samples, isTerminal, conf, last_step);
  }
  Benchmark::close();
  // If required, learn policy from the q_value
  if (conf.policy_samples >= 0)
  {
    Benchmark::open("Policy training set");
    int x_dim = conf.getStateLimits().rows();
    int u_dim = conf.getActionLimits().rows();
    // First generate the starting states
    std::vector<Eigen::VectorXd> states = getPolicyTrainingStates(samples, conf);
    // Then get corresponding actions
    std::vector<Eigen::VectorXd> actions = getPolicyActions(states, conf);
    conf.p_training_set_time += Benchmark::close();

    // Train a policy for each dimension
    policies.clear();
    regression_forests::ExtraTrees policy_learner;
    if (conf.auto_parameters)
    {
      Approximation::ID policy_appr_type = Approximation::ID::PWL;
      // Use GP if required
      if (conf.gp_policies) policy_appr_type = Approximation::ID::GP;
      policy_learner.conf = ExtraTrees::Config::generateAuto(conf.getStateLimits(),
                                                             samples.size(),
                                                             policy_appr_type);
      policy_learner.conf.nb_threads = conf.nb_threads;
      // If using GP, use the custom parameters for hyperparameters tuning
      policy_learner.conf.gp_conf = conf.hyper_rprop_conf;
    }
    else
    {
      policy_learner.conf = conf.policy_conf;
    }
    Benchmark::open("policy_learning");
    /// Create a policy for each action dimension
    for (int dim = 0; dim < u_dim; dim++)
    {
      // First build training set from State to action[dim]
      TrainingSet ts(x_dim);
      for (size_t sample_idx = 0; sample_idx < states.size(); sample_idx++)
      {
        ts.push(regression_forests::Sample(states[sample_idx], actions[sample_idx](dim)));
      }
      policies.push_back(policy_learner.solve(ts, conf.getStateLimits()));
    }
    conf.p_extra_trees_time += Benchmark::close();
  }
}

TrainingSet FPF::getTrainingSet(const std::vector<Sample>& samples,
                                std::function<bool(const Eigen::VectorXd&)> is_terminal,
                                const Config &conf)
{
  std::vector<std::thread> threads;
  std::mutex ts_mutex;
  int x_dim = conf.getStateLimits().rows();
  int u_dim = conf.getActionLimits().rows();
  TrainingSet ts(x_dim + u_dim);
  MultiCore::Intervals intervals = MultiCore::buildIntervals(samples.size(), conf.nb_threads);
  for (size_t thread_no = 0; thread_no < intervals.size(); thread_no++)
  {
    // Compute samples in [start, end[
    int start = intervals[thread_no].first;
    int end = intervals[thread_no].second;
    threads.push_back(std::thread([&, start, end]()
                                  {
                                    TrainingSet thread_ts = this->getTrainingSet(samples,
                                                                                 is_terminal,
                                                                                 conf,
                                                                                 start,
                                                                                 end);
                                    // Only one thread at a time can push its collection
                                    ts_mutex.lock();
                                    for (size_t sample = 0; sample < thread_ts.size(); sample++)
                                    {
                                      ts.push(thread_ts(sample));
                                    }
                                    ts_mutex.unlock();
                                  }));
  }
  for (size_t thread_no = 0; thread_no < intervals.size(); thread_no++)
  {
    threads[thread_no].join();
  }
  return ts;
}

TrainingSet FPF::getTrainingSet(const std::vector<Sample> &samples,
                                std::function<bool(const Eigen::VectorXd&)> is_terminal,
                                const Config &conf,
                                int start_idx, int end_idx)
{
  int x_dim = conf.getStateLimits().rows();
  int u_dim = conf.getActionLimits().rows();
  TrainingSet ls(x_dim + u_dim);
  for (int i = start_idx; i < end_idx; i++) {
    const Sample& sample = samples[i];
    int x_dim = sample.state.rows();
    int u_dim = sample.action.rows();
    Eigen::VectorXd input(x_dim + u_dim);
    input.segment(0, x_dim) = sample.state;
    input.segment(x_dim, u_dim) = sample.action;
    Eigen::VectorXd next_state = sample.next_state;
    double reward = sample.reward;
    if (q_value && !is_terminal(next_state)) {
      // Establishing limits for projection
      Eigen::MatrixXd limits(x_dim + u_dim, 2);
      limits.block(    0, 0, x_dim, 1) = next_state;
      limits.block(    0, 1, x_dim, 1) = next_state;
      limits.block(x_dim, 0, u_dim, 2) = conf.getActionLimits();
      double best_reward;
      if (conf.gp_values) {
        // Preparing functions
        std::function<Eigen::VectorXd(const Eigen::VectorXd)> gradient_func;
        gradient_func = [this](const Eigen::VectorXd & input)
          {
            return this->q_value->getGradient(input);
          };
        std::function<double(const Eigen::VectorXd)> scoring_func;
        scoring_func = [this](const Eigen::VectorXd & guess)
          {
            return this->q_value->getValue(guess);
          };
        // Performing multiple rProp and conserving the best candidate
        Eigen::VectorXd best_guess;
        best_guess = rosban_gp::RandomizedRProp::run(gradient_func, scoring_func,
                                                     limits, conf.find_max_rprop_conf);
        best_reward = scoring_func(best_guess);
      }
      else {
        std::unique_ptr<regression_forests::Tree> sub_tree;
        sub_tree = q_value->unifiedProjectedTree(limits, conf.max_action_tiles);
        best_reward = sub_tree->getMax(limits);
      }
      reward += conf.discount * best_reward;
    }
    ls.push(regression_forests::Sample(input, reward));
  }
  return ls;
}

std::vector<Eigen::VectorXd> FPF::getPolicyTrainingStates(const std::vector<Sample>& samples,
                                                          const Config &conf)
{
  if (conf.policy_samples > 0)
  {
    return rosban_random::getUniformSamples(conf.getStateLimits(), conf.policy_samples);
  }
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples.size());
  for (const Sample & s : samples)
  {
    result.push_back(s.state);
  }
  return result;
}

std::vector<Eigen::VectorXd> FPF::getPolicyActions(const std::vector<Eigen::VectorXd> &states,
                                                   const Config &conf)
{
  MultiCore::Intervals intervals = MultiCore::buildIntervals(states.size(), conf.nb_threads);
  // Each thread has its own vector of actions
  std::vector<std::thread> threads;
  std::vector<std::vector<Eigen::VectorXd>> thread_actions(intervals.size());
  for (size_t thread_no = 0; thread_no < intervals.size(); thread_no++)
  {
    // Compute samples in [start, end[
    int start = intervals[thread_no].first;
    int end = intervals[thread_no].second;
    threads.push_back(std::thread([&, thread_no, start, end]()
                                  {
                                    thread_actions[thread_no]= this->getPolicyActions(states,
                                                                                      conf,
                                                                                      start,
                                                                                      end);
                                  }));
  }
  // Gathering all actions in the right order
  std::vector<Eigen::VectorXd> actions;
  for (size_t thread_no = 0; thread_no < intervals.size(); thread_no++)
  {
    threads[thread_no].join();
    const std::vector<Eigen::VectorXd> & to_add = thread_actions[thread_no];
    actions.insert(actions.end(), to_add.begin(), to_add.end());
  }
  return actions;  
}

std::vector<Eigen::VectorXd> FPF::getPolicyActions(const std::vector<Eigen::VectorXd> &states,
                                                   const Config &conf,
                                                   int start_idx, int end_idx)
{
  int x_dim = conf.getStateLimits().rows();
  int u_dim = conf.getActionLimits().rows();
  std::vector<Eigen::VectorXd> actions;
  for (int i = start_idx; i < end_idx; i++)
  {
    Eigen::MatrixXd limits(x_dim + u_dim, 2);
    limits.block(    0, 0, x_dim, 1) = states[i];
    limits.block(    0, 1, x_dim, 1) = states[i];
    limits.block(x_dim, 0, u_dim, 2) = conf.getActionLimits();
    Eigen::VectorXd best_input;
    if (conf.gp_values) {
      // Preparing functions
      std::function<Eigen::VectorXd(const Eigen::VectorXd)> gradient_func;
      gradient_func = [this](const Eigen::VectorXd & input)
        {
          return this->q_value->getGradient(input);
        };
      std::function<double(const Eigen::VectorXd)> scoring_func;
      scoring_func = [this](const Eigen::VectorXd & guess)
        {
          return this->q_value->getValue(guess);
        };
      // Performing multiple rProp and conserving the best candidate
      best_input = rosban_gp::RandomizedRProp::run(gradient_func, scoring_func,
                                                   limits, conf.find_max_rprop_conf);
    }
    else {
      std::unique_ptr<regression_forests::Tree> sub_tree;
      sub_tree = q_value->unifiedProjectedTree(limits, conf.max_action_tiles);
      best_input = sub_tree->getArgMax(limits);
    }
    actions.push_back(best_input.segment(x_dim, u_dim));
  }

  return actions;
}

}
