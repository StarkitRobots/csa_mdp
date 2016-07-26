#include "rosban_csa_mdp/solvers/learner.h"

namespace csa_mdp
{

Learner::Learner()
  : discount(0.98), nb_threads(1)
{}

Learner::~Learner() {}

void Learner::setStart()
{
  learning_start = rosban_utils::TimeStamp::now();
}

void Learner::setNbThreads(int new_nb_threads)
{
  nb_threads = new_nb_threads;
}

void Learner::setStateLimits(const Eigen::MatrixXd & new_state_limits)
{
  state_limits = new_state_limits;
}

void Learner::setActionLimits(const Eigen::MatrixXd & new_action_limits)
{
  action_limits = new_action_limits;
}

void Learner::setTerminalFunction(std::function<bool(const Eigen::VectorXd &)> is_terminal)
{
  terminal_function = is_terminal;
}

void Learner::setDiscount(double new_discount)
{
  discount = new_discount;
}

Eigen::MatrixXd Learner::getStateLimits() const
{
  return state_limits;
}

Eigen::MatrixXd Learner::getActionLimits() const
{
  return action_limits;
}

void Learner::feed(const csa_mdp::Sample & sample)
{
  samples.push_back(sample);
}


void Learner::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<double>("discount"  , discount  , out);
  rosban_utils::xml_tools::write<int>   ("nb_threads", nb_threads, out);
}


void Learner::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<double>(node, "discount"  , discount  );
  rosban_utils::xml_tools::try_read<int>   (node, "nb_threads", nb_threads);
}

double Learner::getLearningTime() const
{
  return diffSec(learning_start, rosban_utils::TimeStamp::now());
}

const std::map<std::string, double> & Learner::getTimeRepartition() const
{
  return time_repartition;
}

}
