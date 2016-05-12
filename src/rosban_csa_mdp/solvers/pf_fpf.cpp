#include "rosban_csa_mdp/solvers/pf_fpf.h"

#include "rosban_random/tools.h"

#include "rosban_utils/time_stamp.h"

#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

using rosban_utils::TimeStamp;

using regression_forests::ApproximationType;
using regression_forests::ExtraTrees;
using regression_forests::Forest;
using regression_forests::TrainingSet;

namespace csa_mdp
{

PF_FPF::Config::Config()
{
  nb_threads = 1;
  x_dim = 0;
  u_dim = 0;
  horizon = 1;
  discount = 0.99;
  max_action_tiles = 0;
  q_value_time = 0;
  policy_time = 0;
}

const Eigen::MatrixXd & PF_FPF::Config::getStateLimits() const
{
  return x_limits;
}

const Eigen::MatrixXd & PF_FPF::Config::getActionLimits() const
{
  return u_limits;
}

Eigen::MatrixXd PF_FPF::Config::getInputLimits() const
{
  // Init Knownness Forest
  int s_dim = getStateLimits().rows();
  int a_dim = getActionLimits().rows();
  Eigen::MatrixXd limits(s_dim + a_dim, 2);
  limits.block(    0, 0, s_dim, 2) = getStateLimits();
  limits.block(s_dim, 0, a_dim, 2) = getActionLimits();
  return limits;
}

void PF_FPF::Config::setStateLimits(const Eigen::MatrixXd &new_limits)
{
  x_limits = new_limits;
  x_dim = x_limits.rows();
}

void PF_FPF::Config::setActionLimits(const Eigen::MatrixXd &new_limits)
{
  u_limits = new_limits;
  u_dim = u_limits.rows();
}

std::string PF_FPF::Config::class_name() const
{
  return "PF_FPFConfig";
}

void PF_FPF::Config::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("x_dim", x_dim, out);
  rosban_utils::xml_tools::write<int>("u_dim", u_dim, out);
  // Gathering limits in a vector
  std::vector<double> x_limits_vec(x_limits.data(), x_limits.data() + x_limits.size());
  std::vector<double> u_limits_vec(u_limits.data(), u_limits.data() + u_limits.size());
  rosban_utils::xml_tools::write_vector<double>("x_limits", x_limits_vec, out);
  rosban_utils::xml_tools::write_vector<double>("u_limits", u_limits_vec, out);
  // Writing properties
  rosban_utils::xml_tools::write<int>("horizon", horizon, out);
  rosban_utils::xml_tools::write<int>("nb_threads", nb_threads, out);
  rosban_utils::xml_tools::write<double>("discount", discount, out);
  rosban_utils::xml_tools::write<int>("max_action_tiles", max_action_tiles, out);
  rosban_utils::xml_tools::write<double>("q_value_time", q_value_time, out);
  rosban_utils::xml_tools::write<double>("policy_time", policy_time, out);
}

void PF_FPF::Config::from_xml(TiXmlNode *node)
{
  x_dim = rosban_utils::xml_tools::read<int>(node, "x_dim");
  u_dim = rosban_utils::xml_tools::read<int>(node, "u_dim");
  // Gathering limits in a vector
  std::vector<double> x_limits_vec, u_limits_vec;
  x_limits_vec = rosban_utils::xml_tools::read_vector<double>(node, "x_limits");
  u_limits_vec = rosban_utils::xml_tools::read_vector<double>(node, "u_limits");
  if (x_limits_vec.size() != 2 * x_dim)
    throw std::runtime_error("PF_FPF::from_xml: Invalid number of limits for x_limits");
  if (u_limits_vec.size() != 2 * u_dim)
    throw std::runtime_error("PF_FPF::from_xml: Invalid number of limits for x_limits");
  x_limits = Eigen::Map<Eigen::MatrixXd>(x_limits_vec.data(),x_dim, 2);
  u_limits = Eigen::Map<Eigen::MatrixXd>(u_limits_vec.data(),u_dim, 2);
  // Writing properties
  horizon          = rosban_utils::xml_tools::read<int>   (node, "horizon");
  nb_threads       = rosban_utils::xml_tools::read<int>   (node, "nb_threads");
  discount         = rosban_utils::xml_tools::read<double>(node, "discount");
  max_action_tiles = rosban_utils::xml_tools::read<int>   (node, "max_action_tiles");
  q_value_time     = rosban_utils::xml_tools::read<double>(node, "q_value_time");
  policy_time      = rosban_utils::xml_tools::read<double>(node, "policy_time");
}

PF_FPF::PF_FPF()
{
}

TrainingSet PF_FPF::getTrainingSet(const std::vector<Sample> &samples,
                                   std::function<bool(const Eigen::VectorXd&)> is_terminal,
                                   const Config &conf,
                                   std::shared_ptr<regression_forests::Forest> q_value)
{
  std::vector<std::thread> threads;
  std::mutex ts_mutex;
  int x_dim = conf.getStateLimits().rows();
  int u_dim = conf.getActionLimits().rows();
  TrainingSet ts(x_dim + u_dim);
  double samples_by_thread = samples.size() / (double)conf.nb_threads;
  for (int thread_no = 0; thread_no < conf.nb_threads; thread_no++)
  {
    // Compute samples in [start, end[
    int start = std::floor(thread_no * samples_by_thread);
    int end = std::floor((thread_no + 1) * samples_by_thread);
    threads.push_back(std::thread([&]()
                                  {
                                    TrainingSet thread_ts = this->getTrainingSet(samples,
                                                                                 is_terminal,
                                                                                 conf,
                                                                                 q_value,
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
  for (int thread_no = 0; thread_no < conf.nb_threads; thread_no++)
  {
    threads[thread_no].join();
  }
  return ts;
}

TrainingSet PF_FPF::getTrainingSet(const std::vector<Sample>& samples,
                                   std::function<bool(const Eigen::VectorXd&)> is_terminal,
                                   const Config &conf,
                                   std::shared_ptr<regression_forests::Forest> q_value,
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

std::unique_ptr<regression_forests::Forest>
PF_FPF::updateQValue(const std::vector<Sample>& samples,
                     std::function<bool(const Eigen::VectorXd&)> is_terminal,
                     const Config &conf,
                     std::unique_ptr<regression_forests::Forest> q_value,
                     bool final_step)
{
  // Choosing approximation type
  ApproximationType appr_type = ApproximationType::PWC;
  if (final_step)
    appr_type = ApproximationType::PWL;
  // Creating q_learner
  ExtraTrees q_learner;
  q_learner.conf = ExtraTrees::Config::generateAuto(conf.getInputLimits(),
                                                    samples.size(),
                                                    appr_type);
  // Obtaining training set
  TrainingSet ts = getTrainingSet(samples, is_terminal, conf,
                                  std::shared_ptr<Forest>(q_value.release()));
  return q_learner.solve(ts, conf.getInputLimits());
}


std::vector<std::unique_ptr<regression_forests::Forest>>
generatePolicy(const std::vector<Sample>& samples,
               std::unique_ptr<regression_forests::Forest> q_value)
{
  (void) samples;
  std::vector<std::unique_ptr<regression_forests::Forest>> result;
  throw std::logic_error("unimplemented function");
  return result;
}

std::vector<std::vector<std::unique_ptr<regression_forests::Forest>>>
generatePolicies()
{
  throw std::logic_error("unimplemented function");
}

}
