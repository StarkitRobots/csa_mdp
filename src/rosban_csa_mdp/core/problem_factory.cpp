#include "rosban_csa_mdp/core/problem_factory.h"

using csa_mdp::Problem;

namespace csa_mdp
{

std::map<std::string,ProblemFactory::Builder> ProblemFactory::extra_builders;

ProblemFactory::ProblemFactory()
{
  // This factory does not contain default problems
  for (const auto & entry : extra_builders)
  {
    registerBuilder(entry.first, entry.second);
  }
}

void ProblemFactory::registerExtraBuilder(const std::string &name, Builder b)
{
  extra_builders[name] = b;
}

}
