#include "rosban_csa_mdp/core/policy_factory.h"

#include "rosban_csa_mdp/core/forests_policy.h"
#include "rosban_csa_mdp/core/random_policy.h"

using csa_mdp::Policy;
using csa_mdp::ForestsPolicy;
using csa_mdp::RandomPolicy;

std::map<std::string,PolicyFactory::Builder> PolicyFactory::extra_builders;

PolicyFactory::PolicyFactory()
{
  registerBuilder("forests_policy",
                  [](TiXmlNode *node)
                  {
                    ForestsPolicy * p = new ForestsPolicy();
                    p->from_xml(node);
                    return (Policy*)p;
                  });
  registerBuilder("random",
                  [](TiXmlNode *node) {(void)node;return (Policy*)new RandomPolicy();});
  for (const auto & entry : extra_builders)
  {
    registerBuilder(entry.first, entry.second);
  }
}

void PolicyFactory::registerExtraBuilder(const std::string &name, Builder b)
{
  extra_builders[name] = b;
}
