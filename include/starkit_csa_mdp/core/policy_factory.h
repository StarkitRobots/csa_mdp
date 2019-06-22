#pragma once

#include "starkit_csa_mdp/core/policy.h"

#include "starkit_utils/serialization/factory.h"

#include <map>

namespace csa_mdp
{
class PolicyFactory : public starkit_utils::Factory<Policy>
{
public:
  PolicyFactory();

  static void registerExtraBuilder(const std::string& name, Builder b);

private:
  static std::map<std::string, Builder> extra_builders;
};

}  // namespace csa_mdp
