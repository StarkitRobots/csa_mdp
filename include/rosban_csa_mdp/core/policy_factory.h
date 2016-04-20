#pragma once

#include "rosban_csa_mdp/core/policy.h"

#include "rosban_utils/factory.h"

#include <map>

class PolicyFactory : public rosban_utils::Factory<csa_mdp::Policy>
{
public:

  PolicyFactory();

  static void registerExtraBuilder(const std::string &name, Builder b);

private:
  static std::map<std::string,Builder> extra_builders;

};
