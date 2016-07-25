#pragma once

#include "rosban_csa_mdp/core/problem.h"

#include "rosban_utils/factory.h"

#include <map>

namespace csa_mdp
{

class ProblemFactory : public rosban_utils::Factory<Problem>
{
public:

  ProblemFactory();

  static void registerExtraBuilder(const std::string &name, Builder b, bool parse_xml = true);
  static void registerExtraBuilder(const std::string &name, XMLBuilder b);

private:
  static std::map<std::string, XMLBuilder> extra_builders;
};

}
