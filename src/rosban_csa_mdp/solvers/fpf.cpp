#include "rosban_csa_mdp/solvers/fpf.h"

#include "rosban_random/tools.h"

#include "rosban_utils/benchmark.h"
#include "rosban_utils/multi_core.h"

#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

using rosban_utils::Benchmark;
using rosban_utils::MultiCore;
using rosban_utils::TimeStamp;

using regression_forests::ApproximationType;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;


using namespace rosban_utils::xml_tools;//Read and writes

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

std::string FPF::Config::class_name() const
{
  return "FPFConfig";
}

void FPF::Config::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("x_dim", x_dim, out);
  rosban_utils::xml_tools::write<int>("u_dim", u_dim, out);
  // Gathering limits in a vector if dim have been specified
  std::vector<double> x_limits_vec(x_limits.data(), x_limits.data() + x_limits.size());
  std::vector<double> u_limits_vec(u_limits.data(), u_limits.data() + u_limits.size());
  rosban_utils::xml_tools::write_vector<double>("x_limits", x_limits_vec, out);
  rosban_utils::xml_tools::write_vector<double>("u_limits", u_limits_vec, out);
  // Writing properties
  rosban_utils::xml_tools::write<int>("horizon", horizon, out);
  rosban_utils::xml_tools::write<int>("nb_threads", nb_threads, out);
  rosban_utils::xml_tools::write<double>("discount", discount, out);
  rosban_utils::xml_tools::write<int>("policy_samples", policy_samples, out);
  rosban_utils::xml_tools::write<int>("max_action_tiles", max_action_tiles, out);
  rosban_utils::xml_tools::write<bool>("auto_parameters", auto_parameters, out);
  if (!auto_parameters)
  {
    q_value_conf.write("q_value_conf", out);
    policy_conf.write("policy_conf", out);
  }
}

void FPF::Config::from_xml(TiXmlNode *node)
{
  // Reading size of the problem if provided
  rosban_utils::xml_tools::try_read<int>(node, "x_dim", x_dim);
  rosban_utils::xml_tools::try_read<int>(node, "u_dim", u_dim);
  // Gathering limits in a vector
  if (x_dim != 0)
  {
    std::vector<double> x_limits_vec;
    x_limits_vec = rosban_utils::xml_tools::read_vector<double>(node, "x_limits");
    if (x_limits_vec.size() != (size_t) 2 * x_dim)
      throw std::runtime_error("FPF::from_xml: Invalid number of limits for x_limits");
    x_limits = Eigen::Map<Eigen::MatrixXd>(x_limits_vec.data(),x_dim, 2);
  }
  if (u_dim != 0)
  {
    std::vector<double> u_limits_vec;
    u_limits_vec = rosban_utils::xml_tools::read_vector<double>(node, "u_limits");
    if (u_limits_vec.size() != (size_t)2 * u_dim)
      throw std::runtime_error("FPF::from_xml: Invalid number of limits for u_limits");
    u_limits = Eigen::Map<Eigen::MatrixXd>(u_limits_vec.data(),u_dim, 2);
  }
  // Reading mandatory properties
  horizon          = rosban_utils::xml_tools::read<int>   (node, "horizon"         );
  discount         = rosban_utils::xml_tools::read<double>(node, "discount"        );
  max_action_tiles = rosban_utils::xml_tools::read<int>   (node, "max_action_tiles");
  // Reading optional properties
  rosban_utils::xml_tools::try_read<int>   (node, "nb_threads"      , nb_threads      );
  rosban_utils::xml_tools::try_read<int>   (node, "policy_samples"  , policy_samples  );
  rosban_utils::xml_tools::try_read<bool>  (node, "auto_parameters" , auto_parameters );
  if (!auto_parameters)
  {
    q_value_conf.read(node, "q_value_conf");
    policy_conf.read(node, "policy_conf");
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
    ApproximationType appr_type = last_step ? ApproximationType::PWL : ApproximationType::PWC;
    // TODO implement alternative update with PWL approximations to allow their use
    appr_type = ApproximationType::PWC;
    // Generating the configuration automatically
    q_learner.conf = ExtraTrees::Config::generateAuto(conf.getInputLimits(),
                                                      samples.size(),
                                                      appr_type);
    q_learner.conf.nb_threads = conf.nb_threads;
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
      policy_learner.conf = ExtraTrees::Config::generateAuto(conf.getStateLimits(),
                                                             samples.size(),
                                                             ApproximationType::PWL);
      policy_learner.conf.nb_threads = conf.nb_threads;
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
      std::unique_ptr<regression_forests::Tree> sub_tree;
      sub_tree = q_value->unifiedProjectedTree(limits, conf.max_action_tiles);
      reward += conf.discount * sub_tree->getMax(limits);
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
    std::unique_ptr<regression_forests::Tree> sub_tree;
    sub_tree = q_value->unifiedProjectedTree(limits, conf.max_action_tiles);
    actions.push_back(sub_tree->getArgMax(limits).segment(x_dim, u_dim));
  }

  return actions;
}

}
