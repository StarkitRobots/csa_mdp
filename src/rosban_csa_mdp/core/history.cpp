#include "rosban_csa_mdp/core/history.h"
#include "rosban_csa_mdp/core/problem_factory.h"

#include "rosban_utils/string_tools.h"

#include <fstream>
#include <sstream>

namespace csa_mdp
{

History::Config::Config()
  : run_column(-1), step_column(-1)
{
}

void History::Config::to_xml(std::ostream &out) const
{
  if (!problem)
  {
    throw std::runtime_error("History::Config::to_xml: forbidden while problem is not set");
  }
  out << "<problem>";
  problem->write(problem->class_name(), out);
  out << "</problem>";
  rosban_utils::xml_tools::write<std::string>("log_path"      , log_path      , out);
  rosban_utils::xml_tools::write<int>        ("run_column"    , run_column    , out);
  rosban_utils::xml_tools::write<int>        ("step_column"   , step_column   , out);
  rosban_utils::xml_tools::write_vector<int> ("state_columns" , state_columns , out);
  rosban_utils::xml_tools::write_vector<int> ("action_columns", action_columns, out);
}

void History::Config::from_xml(TiXmlNode *node)
{
  // Try to read problem if found
  TiXmlNode * problem_node = node->FirstChild("problem");
  if(problem_node)
  {
    problem = std::unique_ptr<Problem>(ProblemFactory().build(problem_node));
  }

  rosban_utils::xml_tools::try_read<std::string>(node, "log_path"      , log_path      );
  rosban_utils::xml_tools::try_read<int>        (node, "run_column"    , run_column    );
  rosban_utils::xml_tools::try_read<int>        (node, "step_column"   , step_column   );
  rosban_utils::xml_tools::try_read_vector<int> (node, "state_columns" , state_columns );
  rosban_utils::xml_tools::try_read_vector<int> (node, "action_columns", action_columns);
}

std::string History::Config::class_name() const
{
  return "history_config";
}


History::History()
{
}

void History::push(const Eigen::VectorXd &state,
                   const Eigen::VectorXd &action,
                   double reward)
{
  states.push_back(state);
  actions.push_back(action);
  rewards.push_back(reward);
}

std::vector<Sample> History::getBatch() const
{
  std::vector<Sample> samples;
  if (states.size() == 0) return samples;
  for (size_t i = 0; i < states.size() - 1; i++)
  {
    samples.push_back(Sample(states[i], actions[i], states[i+1], rewards[i + 1]));
  }
  return samples;
}

std::vector<Sample> History::getBatch(const std::vector<History> &histories)
{
  std::vector<Sample> samples;
  for (const History &h : histories)
  {
    std::vector<Sample> new_samples = h.getBatch();
    samples.reserve(samples.size() + new_samples.size());
    std::move(new_samples.begin(), new_samples.end(), std::inserter(samples, samples.end()));
  }
  return samples;
}

std::vector<History> History::readCSV(const History::Config & conf)
{
  if (conf.log_path == "")
  {
    throw std::runtime_error("History::readCSV: Trying to read from a conf with log_path=\"\"");
  }
  // TODO start by validating config and accept other format
  int nb_states = conf.problem->getStateLimits().rows();
  int nb_actions = conf.problem->getActionLimits().rows();
  return readCSV(conf.log_path, nb_states, nb_actions);
}

std::vector<History> History::readCSV(const std::string &path,
                                      int nb_states,
                                      int nb_actions)
{
  std::vector<int> state_cols, action_cols;
  for (int i = 0; i < nb_states; i++)
  {
    state_cols.push_back(i+2);
  }
  for (int i = 0; i < nb_actions; i++)
  {
    action_cols.push_back(i+2+nb_states);
  }
  int reward_col = 2 + nb_states + nb_actions;
  return readCSV(path, 0, 1, state_cols, action_cols, reward_col, true);
}

std::vector<History> History::readCSV(const std::string &path,
                                      int run_col,
                                      int step_col,
                                      const std::vector<int> &state_cols,
                                      const std::vector<int> &action_cols,
                                      int reward_col,
                                      bool header)
{
  std::vector<History> histories;
  int x_dim = state_cols.size();
  int u_dim = action_cols.size();
  std::ifstream infile(path);
  if (!infile.good())
  {
    throw std::runtime_error("History::readCSV: Failed to open file '" + path + "'");
  }
  std::string line;
  // Temporary variables
  History curr_history;
  int curr_run = 1;
  int expected_step = 0;
  while (std::getline(infile, line))
  {
    // Skipping first line if needed
    if (header)
    {
      header = false;
      continue;
    }
    // Getting columns
    std::vector<std::string> cols = rosban_utils::split_string(line,',');
    // Checking if a new run is started (if run-col is specified)
    if (run_col >= 0 && std::stoi(cols[run_col]) != curr_run)
    {
      histories.push_back(curr_history);
      curr_run = std::stoi(cols[run_col]);
      curr_history = History();
      expected_step = 0;
    }
    // Checking
    if (step_col >= 0 && std::stoi(cols[step_col]) != expected_step)
    {
      std::ostringstream oss;
      oss << "Unexpected step: '" << cols[step_col] << "' expecting step '"
          << expected_step << "'";
      throw std::runtime_error(oss.str());
    }

    // Declaring variables
    Eigen::VectorXd state(x_dim);
    Eigen::VectorXd action(u_dim);
    double reward = 0;
    // Attributing variables at the right place
    int dim = 0;
    for (int col : state_cols)
    {
      if (col >= (int)cols.size())
        throw std::runtime_error("Not enough columns in logs");
      state(dim++) = std::stod(cols[col]);
    }
    dim = 0;
    for (int col : action_cols)
    {
      if (col >= (int)cols.size())
        throw std::runtime_error("Not enough columns in logs");
      action(dim++) = std::stod(cols[col]);
    }
    if (reward_col >= 0)
    {
      if (reward_col >= (int)cols.size())
        throw std::runtime_error("Not enough columns in logs");
      reward = std::stod(cols[reward_col]);
    }
    // Pushing values
    curr_history.push(state, action, reward);
    expected_step++;
  }
  histories.push_back(curr_history);
  return histories;
}

std::vector<History> History::readCSV(const std::string &path,
                                      int run_col,
                                      int step_col,
                                      const std::vector<int> &state_cols,
                                      const std::vector<int> &action_cols,
                                      Problem::RewardFunction compute_reward,
                                      bool header)
{
  // Factorizing code
  std::vector<History> histories = History::readCSV(path, run_col, step_col,
                                                    state_cols, action_cols, -1, header);
  // Computing rewards
  for (auto & h : histories)
  {
    for (size_t i = 1; i < h.states.size(); i++)
    {
      h.rewards[i] = compute_reward(h.states[i-1], h.actions[i-1], h.states[i]);
    }
  }
  return histories;
}

History History::readCSV(const std::string &path,
                         const std::vector<int> &state_cols,
                         const std::vector<int> &action_cols,
                         int reward_col,
                         bool header)
{
  return readCSV(path, -1, -1, state_cols, action_cols, reward_col, header)[0];
}

History History::readCSV(const std::string &path,
                         const std::vector<int> &state_cols,
                         const std::vector<int> &action_cols,
                         Problem::RewardFunction compute_reward,
                         bool header)
{
  return readCSV(path, -1, -1, state_cols, action_cols, compute_reward, header)[0];
}

}
