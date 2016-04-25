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

  static void registerExtraBuilder(const std::string &name, Builder b);

private:
  static std::map<std::string,Builder> extra_builders;
};

}
