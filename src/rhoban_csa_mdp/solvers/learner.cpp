#include "rhoban_csa_mdp/solvers/learner.h"

namespace csa_mdp
{
Learner::Learner() : discount(0.98), nb_threads(1)
{
}

Learner::~Learner()
{
}

void Learner::setStart()
{
  learning_start = rhoban_utils::TimeStamp::now();
}

void Learner::setNbThreads(int new_nb_threads)
{
  nb_threads = new_nb_threads;
}

void Learner::endRun()
{
}

void Learner::setStateLimits(const Eigen::MatrixXd& new_state_limits)
{
  state_limits = new_state_limits;
}

void Learner::setActionLimits(const std::vector<Eigen::MatrixXd>& new_action_limits)
{
  action_limits = new_action_limits;
}

void Learner::setDiscount(double new_discount)
{
  discount = new_discount;
}

Eigen::MatrixXd Learner::getStateLimits() const
{
  return state_limits;
}

std::vector<Eigen::MatrixXd> Learner::getActionLimits() const
{
  return action_limits;
}

void Learner::feed(const csa_mdp::Sample& sample)
{
  samples.push_back(sample);
}

Json::Value Learner::toJson() const
{
  Json::Value v;
  v["discount"] = discount;
  v["nb_threads"] = nb_threads;
  return v;
}

void Learner::fromJson(const Json::Value& v, const std::string& dir_name)
{
  (void)dir_name;
  rhoban_utils::tryRead(v, "discount", &discount);
  rhoban_utils::tryRead(v, "nb_threads", &nb_threads);
}

double Learner::getLearningTime() const
{
  return diffSec(learning_start, rhoban_utils::TimeStamp::now());
}

const std::map<std::string, double>& Learner::getTimeRepartition() const
{
  return time_repartition;
}

}  // namespace csa_mdp
