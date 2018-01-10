#pragma once

#include "rosban_csa_mdp/core/policy.h"

#include "rhoban_utils/serialization/factory.h"

#include <map>

namespace csa_mdp
{

class PolicyFactory : public rhoban_utils::Factory<Policy>
{
public:

  PolicyFactory();

  static void registerExtraBuilder(const std::string &name, Builder b);

private:
  static std::map<std::string,Builder> extra_builders;

};

}
