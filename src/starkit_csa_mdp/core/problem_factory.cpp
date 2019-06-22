#include "starkit_csa_mdp/core/problem_factory.h"

using csa_mdp::Problem;

namespace csa_mdp
{
std::map<std::string, ProblemFactory::JsonBuilder> ProblemFactory::extra_builders;

ProblemFactory::ProblemFactory()
{
  // This factory does not contain default problems
  for (const auto& entry : extra_builders)
  {
    registerBuilder(entry.first, entry.second);
  }
}

void ProblemFactory::registerExtraBuilder(const std::string& name, Builder b, bool parse_json)
{
  registerExtraBuilder(name, toJsonBuilder(b, parse_json));
}

void ProblemFactory::registerExtraBuilder(const std::string& name, JsonBuilder b)
{
  extra_builders[name] = b;
}

}  // namespace csa_mdp
