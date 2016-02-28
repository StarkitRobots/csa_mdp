#include "rosban_csa_mdp/solvers/fpf.h"

#include "rosban_regression_forests/tools/random.h"

#include "rosban_utils/time_stamp.h"

#include <iostream>
#include <sstream>

using rosban_utils::TimeStamp;

using regression_forests::ApproximationType;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;


using namespace rosban_utils::xml_tools;//Read and writes

namespace csa_mdp
{

FPF::Config::Config()
{
  x_dim = 0;
  u_dim = 0;
  horizon = 1;
  discount = 0.99;
  max_action_tiles = 0;
  policy_samples = 0;
  q_value_time = 0;
  policy_time = 0;
}

std::vector<std::string> FPF::Config::names() const
{
  std::vector<std::string> result = {"x_dim", "u_dim"};
  for (size_t i = 0; i < x_dim; i++)
  {
    result.push_back("x_" + std::to_string(i) + "_min");
    result.push_back("x_" + std::to_string(i) + "_max");
  }
  for (size_t i = 0; i < u_dim; i++)
  {
    result.push_back("u_" + std::to_string(i) + "_min");
    result.push_back("u_" + std::to_string(i) + "_max");
  }
  result.push_back("horizon"         );
  result.push_back("discount"        );
  result.push_back("max_action_tiles");
  result.push_back("policy_samples"  );
  result.push_back("q_value_time"    );
  result.push_back("policy_time"     );

  // Add q_value_conf with prefix
  std::vector<std::string> q_value_names = q_value_conf.names();
  for (const std::string &name : q_value_names)
    result.push_back("q_value_" + name);
  std::vector<std::string> policy_names = policy_conf.names();
  for (const std::string &name : policy_names)
    result.push_back("policy_" + name);
  return result;
}

std::vector<std::string> FPF::Config::values() const
{
  std::vector<std::string> result = {std::to_string(x_dim), std::to_string(u_dim)};
  // Use custom to_string for double here
  for (size_t i = 0; i < x_dim; i++)
  {
    result.push_back(std::to_string(x_limits(i,0)));
    result.push_back(std::to_string(x_limits(i,1)));
  }
  for (size_t i = 0; i < u_dim; i++)
  {
    result.push_back(std::to_string(u_limits(i,0)));
    result.push_back(std::to_string(u_limits(i,1)));
  }
  result.push_back(std::to_string(horizon         ));
  result.push_back(std::to_string(discount        ));
  result.push_back(std::to_string(max_action_tiles));
  result.push_back(std::to_string(policy_samples  ));
  result.push_back(std::to_string(q_value_time    ));
  result.push_back(std::to_string(policy_time     ));
  std::vector<std::string> q_value_values = q_value_conf.values();
  result.insert(result.end(), q_value_values.begin(), q_value_values.end());
  std::vector<std::string> policy_values = policy_conf.values();
  result.insert(result.end(), policy_values.begin(), policy_values.end());
  return result;
}

void FPF::Config::load(const std::vector<std::string>& col_names,
                       const std::vector<std::string>& col_values)
{
  // start by reading the 2 first entries
  if (col_names.size() < 2)
  {
    throw std::runtime_error("Failed to load FPF::Config, size must be at least 2");
  }
  x_dim = std::stoi(col_values[0]);
  u_dim = std::stoi(col_values[1]);
  // Now names will be valid
  auto expectedNames = names();
  if (col_names.size() != expectedNames.size()) {
    throw std::runtime_error("Failed to load FPF::Config, unexpected size");
  }
  // Check if names matches the description
  for (size_t colNo = 0;  colNo < col_names.size(); colNo++) {
    auto givenName = col_names[colNo];
    auto expectedName = expectedNames[colNo];
    if (givenName.find(expectedName) == std::string::npos) {
      throw std::runtime_error("Given name '" + givenName + "' does not match '"
                               + expectedName + "'");
    }
  }
  // Read limits
  int col_idx = 2;
  for (size_t i = 0; i < x_dim; i++)
  {
    x_limits(i,0) = std::stoi(col_values[col_idx++]);
    x_limits(i,1) = std::stoi(col_values[col_idx++]);
  }
  for (size_t i = 0; i < u_dim; i++)
  {
    u_limits(i,0) = std::stoi(col_values[col_idx++]);
    u_limits(i,1) = std::stoi(col_values[col_idx++]);
  }
  horizon          = std::stoi(col_values[col_idx++]);
  discount         = std::stoi(col_values[col_idx++]);
  max_action_tiles = std::stoi(col_values[col_idx++]);
  q_value_time     = std::stod(col_values[col_idx++]);
  policy_time      = std::stod(col_values[col_idx++]);
  /// reading q_value_conf
  int q_conf_size = q_value_conf.names().size();
  std::vector<std::string> q_value_names, q_value_values;
  q_value_names.insert (q_value_names.end() ,
                        col_names.begin() + col_idx ,
                        col_names.begin() + col_idx + q_conf_size);
  q_value_values.insert(q_value_values.end(),
                        col_values.begin() + col_idx,
                        col_values.begin() + col_idx + q_conf_size);
  q_value_conf.load(q_value_names, q_value_values);
  col_idx += q_conf_size;
  /// reading policy_conf
  int policy_conf_size = policy_conf.names().size();
  std::vector<std::string> policy_names, policy_values;
  policy_names.insert (policy_names.end() ,
                       col_names.begin() + col_idx ,
                       col_names.begin() + col_idx + policy_conf_size);
  policy_values.insert(policy_values.end(),
                       col_values.begin() + col_idx,
                       col_values.begin() + col_idx + policy_conf_size);
  policy_conf.load(policy_names, policy_values);
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
  // Gathering limits in a vector
  std::vector<double> x_limits_vec(x_limits.data(), x_limits.data() + x_limits.size());
  std::vector<double> u_limits_vec(u_limits.data(), u_limits.data() + u_limits.size());
  rosban_utils::xml_tools::write_vector<double>("x_limits", x_limits_vec, out);
  rosban_utils::xml_tools::write_vector<double>("u_limits", u_limits_vec, out);
  // Writing properties
  rosban_utils::xml_tools::write<int>("horizon", horizon, out);
  rosban_utils::xml_tools::write<double>("discount", discount, out);
  rosban_utils::xml_tools::write<int>("policy_samples", policy_samples, out);
  rosban_utils::xml_tools::write<int>("max_action_tiles", max_action_tiles, out);
  rosban_utils::xml_tools::write<double>("q_value_time", q_value_time, out);
  rosban_utils::xml_tools::write<double>("policy_time", policy_time, out);
  q_value_conf.write("q_value_conf", out);
  policy_conf.write("policy_conf", out);
}

void FPF::Config::from_xml(TiXmlNode *node)
{
  x_dim = rosban_utils::xml_tools::read<int>(node, "x_dim");
  u_dim = rosban_utils::xml_tools::read<int>(node, "u_dim");
  // Gathering limits in a vector
  std::vector<double> x_limits_vec, u_limits_vec;
  x_limits_vec = rosban_utils::xml_tools::read_vector<double>(node, "x_limits");
  u_limits_vec = rosban_utils::xml_tools::read_vector<double>(node, "u_limits");
  if (x_limits_vec.size() != 2 * x_dim)
    throw std::runtime_error("FPF::from_xml: Invalid number of limits for x_limits");
  if (u_limits_vec.size() != 2 * u_dim)
    throw std::runtime_error("FPF::from_xml: Invalid number of limits for x_limits");
  x_limits = Eigen::Map<Eigen::MatrixXd>(x_limits_vec.data(),x_dim, 2);
  u_limits = Eigen::Map<Eigen::MatrixXd>(u_limits_vec.data(),u_dim, 2);
  // Writing properties
  horizon          = rosban_utils::xml_tools::read<int>   (node, "horizon");
  discount         = rosban_utils::xml_tools::read<double>(node, "discount");
  policy_samples   = rosban_utils::xml_tools::read<int>   (node, "policy_samples");
  max_action_tiles = rosban_utils::xml_tools::read<int>   (node, "max_action_tiles");
  q_value_time     = rosban_utils::xml_tools::read<double>(node, "q_value_time");
  policy_time      = rosban_utils::xml_tools::read<double>(node, "policy_time");
  q_value_conf.read(node, "q_value_conf");
  policy_conf.read(node, "policy_conf");
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
                       std::function<bool(const Eigen::VectorXd&)> isTerminal)
{
  regression_forests::ExtraTrees q_learner;
  q_learner.conf = conf.q_value_conf;
  // ts is computed using last q_value
  TrainingSet ts = getTrainingSet(samples, isTerminal);
  q_value = q_learner.solve(ts, conf.getInputLimits());
}

void FPF::solve(const std::vector<Sample>& samples,
                std::function<bool(const Eigen::VectorXd&)> isTerminal)
{
  q_value.release();
  TimeStamp q_value_start = TimeStamp::now();
  for (size_t h = 1; h <= conf.horizon; h++) {
    updateQValue(samples, isTerminal);
  }
  TimeStamp q_value_end = TimeStamp::now();
  conf.q_value_time = diffSec(q_value_start, q_value_end);
  // If required, learn policy from the q_value
  if (conf.policy_samples >= 0)
  {
    int x_dim = conf.getStateLimits().rows();
    int u_dim = conf.getActionLimits().rows();
    // First generate the starting states
    std::vector<Eigen::VectorXd> states = getPolicyTrainingStates(samples);
    // Then get corresponding actions
    std::vector<Eigen::VectorXd> actions;
    for (const Eigen::VectorXd &state : states)
    {
      // Establishing limits for projection
      Eigen::MatrixXd limits(x_dim + u_dim, 2);
      limits.block(    0, 0, x_dim, 1) = state;
      limits.block(    0, 1, x_dim, 1) = state;
      limits.block(x_dim, 0, u_dim, 2) = conf.getActionLimits();
      std::unique_ptr<regression_forests::Tree> sub_tree;
      sub_tree = q_value->unifiedProjectedTree(limits, conf.max_action_tiles);
      actions.push_back(sub_tree->getArgMax(limits).segment(x_dim, u_dim));
    }
    // Train a policy for each dimension
    policies.clear();
    regression_forests::ExtraTrees policy_learner;
    policy_learner.conf = conf.policy_conf;
    for (int dim = 0; dim < u_dim; dim++)
    {
      // First build training set
      TrainingSet ts(x_dim);
      for (size_t sample_idx = 0; sample_idx < states.size(); sample_idx++)
      {
        ts.push(regression_forests::Sample(states[sample_idx], actions[sample_idx](dim)));
      }
      policies.push_back(policy_learner.solve(ts, conf.getStateLimits()));
    }
    TimeStamp policy_end = TimeStamp::now();
    conf.policy_time = diffSec(q_value_end, policy_end);
  }
}

TrainingSet FPF::getTrainingSet(const std::vector<Sample>& samples,
                                std::function<bool(const Eigen::VectorXd&)> is_terminal)
{
  int x_dim = conf.getStateLimits().rows();
  int u_dim = conf.getActionLimits().rows();
  TrainingSet ls(x_dim + u_dim);
  for (size_t i = 0; i < samples.size(); i++) {
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

std::vector<Eigen::VectorXd> FPF::getPolicyTrainingStates(const std::vector<Sample>& samples)
{
  if (conf.policy_samples > 0)
  {
    return regression_forests::getUniformSamples(conf.getStateLimits(), conf.policy_samples);
  }
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples.size());
  for (const Sample & s : samples)
  {
    result.push_back(s.state);
  }
  return result;
}

}
