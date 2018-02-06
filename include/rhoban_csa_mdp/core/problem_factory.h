#pragma once

#include "rhoban_csa_mdp/core/problem.h"

#include "rhoban_utils/serialization/factory.h"

#include <map>

namespace csa_mdp
{

class ProblemFactory : public rhoban_utils::Factory<Problem>
{
public:

  ProblemFactory();

  static void registerExtraBuilder(const std::string &name, Builder b, bool parse_json = true);
  static void registerExtraBuilder(const std::string &name, JsonBuilder b);

private:
  static std::map<std::string, JsonBuilder> extra_builders;
};

}
