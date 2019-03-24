#include "rhoban_csa_mdp/value_approximators/value_approximator.h"

#include "rhoban_utils/io_tools.h"

namespace csa_mdp
{
ValueApproximator::ValueApproximator() : nb_threads(1)
{
}

ValueApproximator::~ValueApproximator()
{
}

void ValueApproximator::setNbThreads(int nb_threads_)
{
  nb_threads = nb_threads_;
}

Json::Value ValueApproximator::toJson() const
{
  Json::Value v;
  v["nb_threads"] = nb_threads;
  return v;
}

void ValueApproximator::fromJson(const Json::Value& v, const std::string& dir_name)
{
  (void)dir_name;
  rhoban_utils::tryRead(v, "nb_threads", &nb_threads);
}

}  // namespace csa_mdp
