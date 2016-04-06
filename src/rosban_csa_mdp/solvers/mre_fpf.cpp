#include "rosban_csa_mdp/solvers/mre_fpf.h"

#include "rosban_regression_forests/approximations/pwc_approximation.h"

#include "rosban_utils/time_stamp.h"

using rosban_utils::TimeStamp;

using regression_forests::PWCApproximation;
using regression_forests::TrainingSet;

namespace csa_mdp
{

MREFPF::Config::Config()
  : filter_samples(false), reward_max(0), update_type(UpdateType::MRE)
{
}

std::string MREFPF::Config::class_name() const
{
  return "MREFPFConfig";
}

void MREFPF::Config::to_xml(std::ostream &out) const
{
  FPF::Config::to_xml(out);
  rosban_utils::xml_tools::write<bool>  ("filter_samples", filter_samples, out);
  rosban_utils::xml_tools::write<double>("reward_max"    , reward_max    , out);
  rosban_utils::xml_tools::write<std::string>("update_type", to_string(update_type), out);
}

void MREFPF::Config::from_xml(TiXmlNode *node)
{
  FPF::Config::from_xml(node);
  // Mandatory parameters
  reward_max      = rosban_utils::xml_tools::read<int> (node, "reward_max"    );
  std::string update_type_str;
  update_type_str = rosban_utils::xml_tools::read<std::string>(node, "update_type");
  update_type = loadUpdateType(update_type_str);
  // Optional parameters
  rosban_utils::xml_tools::try_read<bool>(node, "filter_samples", filter_samples);
}

MREFPF::MREFPF()
{
}

MREFPF::MREFPF(std::shared_ptr<KnownnessFunction> knownness_func_)
  : knownness_func(knownness_func_)
{
}

TrainingSet MREFPF::getTrainingSet(const std::vector<Sample> &samples,
                                   std::function<bool(const Eigen::VectorXd&)> is_terminal,
                                   const FPF::Config &conf_fpf)
{
  const MREFPF::Config &conf = dynamic_cast<const MREFPF::Config &>(conf_fpf);
  // Removing samples which have the same starting state if filter_samples is activated
  TimeStamp start_filter = TimeStamp::now();
  std::vector<Sample> filtered_samples;
  if (conf.filter_samples)
    filtered_samples = filterSimilarSamples(samples);
  else
    filtered_samples = samples;
  TimeStamp end_filter = TimeStamp::now();
  //std::cout << "\t\tFiltering samples  : " << diffMs(start_filter, end_filter) << " ms" << std::endl;
  // Computing original training Set
  TrainingSet original_ts = FPF::getTrainingSet(filtered_samples, is_terminal, conf);
  TimeStamp get_ts_end = TimeStamp::now();
  //std::cout << "\t\tFPF::getTrainingSet: " << diffMs(end_filter, get_ts_end) << " ms" << std::endl;
  // If alternative mode, then do not modify samples
  if (conf.update_type == UpdateType::Alternative) return original_ts;
  // Otherwise use knownness to influence samples
  TrainingSet new_ts(original_ts.getInputDim());
  for (size_t i = 0; i < original_ts.size(); i++)
  {
    // Extracting information from original sample
    const regression_forests::Sample & original_sample = original_ts(i);
    Eigen::VectorXd input = original_sample.getInput();
    double reward         = original_sample.getOutput();
    // Getting knownness of the input
    double knownness = knownness_func->getValue(input);
    double new_reward = reward * knownness + conf.reward_max * (1 - knownness);
    new_ts.push(regression_forests::Sample(input, new_reward));
  }
  return new_ts;
}


void MREFPF::updateQValue(const std::vector<Sample> &samples,
                          std::function<bool(const Eigen::VectorXd&)> is_terminal,
                          const FPF::Config &conf_fpf,
                          bool last_step)
{
  const MREFPF::Config &conf = dynamic_cast<const MREFPF::Config &>(conf_fpf);
  FPF::updateQValue(samples, is_terminal, conf, last_step);
  if (conf.update_type == UpdateType::Alternative)
  {
    regression_forests::Node::Function f = [this, &conf](regression_forests::Node * node,
                                                         const Eigen::MatrixXd & limits)
      {
        // Throw an error if approximation is not PWC
        PWCApproximation *pwc_app = dynamic_cast<PWCApproximation*>(node->a);
        if (pwc_app == nullptr)
        {
          throw std::logic_error("Alternative update is only available for pwc approximations");
        }
        // Compute knownness and old value
        Eigen::VectorXd middle_point = (limits.col(1) + limits.col(0)) / 2;
        double knownness = this->knownness_func->getValue(middle_point);
        double old_val = pwc_app->getValue();
        double new_val = old_val * knownness + (1 - knownness) * conf.reward_max;
        node->a = new PWCApproximation(new_val);
        delete(pwc_app);
      };
    Eigen::MatrixXd limits = conf.getInputLimits();
    q_value->applyOnLeafs(limits, f);
  }
}

std::vector<Sample> MREFPF::filterSimilarSamples(const std::vector<Sample> &samples) const
{
  std::vector<Sample> filtered_samples;
  for (const Sample &new_sample : samples)
  {
    bool found_similar = false;
    double tolerance = std::pow(10,-6);
    for (const Sample &known_sample : filtered_samples)
    {
      Eigen::VectorXd state_diff = known_sample.state - new_sample.state;
      Eigen::VectorXd action_diff = known_sample.action - new_sample.action;
      // Which is the highest difference between the two samples?
      double max_diff = std::max(state_diff.lpNorm<Eigen::Infinity>(),
                                 action_diff.lpNorm<Eigen::Infinity>());
      if (max_diff < tolerance)
      {
        found_similar = true;
        break;
      }
    }
    // Do not add similar samples
    if (found_similar) continue;
    filtered_samples.push_back(new_sample);
  }
  return filtered_samples;
}

std::string to_string(MREFPF::UpdateType type)
{
  switch (type)
  {
    case MREFPF::UpdateType::MRE: return "MRE";
    case MREFPF::UpdateType::Alternative: return "Alternative";
  }
  throw std::runtime_error("Unknown type in to_string(Type)");
}

MREFPF::UpdateType loadUpdateType(const std::string &type)
{
  if (type == "MRE")
  {
    return MREFPF::UpdateType::MRE;
  }
  if (type == "Alternative")
  {
    return MREFPF::UpdateType::Alternative;
  }
  throw std::runtime_error("Unknown MREFPF Update Type: '" + type + "'");
}

}
