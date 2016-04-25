#pragma once

#include "rosban_csa_mdp/core/policy.h"

#include "rosban_utils/factory.h"

#include <map>

namespace csa_mdp
{

class PolicyFactory : public rosban_utils::Factory<Policy>
{
public:

  PolicyFactory();

  static void registerExtraBuilder(const std::string &name, Builder b);

private:
  static std::map<std::string,Builder> extra_builders;

};

}
