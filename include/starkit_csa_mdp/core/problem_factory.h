#pragma once

#include "starkit_csa_mdp/core/problem.h"

#include "starkit_utils/serialization/factory.h"

#include <map>

namespace csa_mdp
{
class ProblemFactory : public starkit_utils::Factory<Problem>
{
public:
  ProblemFactory();

  static void registerExtraBuilder(const std::string& name, Builder b, bool parse_json = true);
  static void registerExtraBuilder(const std::string& name, JsonBuilder b);

private:
  static std::map<std::string, JsonBuilder> extra_builders;
};

}  // namespace csa_mdp
