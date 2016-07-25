#include "rosban_csa_mdp/core/random_policy.h"

#include "rosban_random/tools.h"

namespace csa_mdp
{

RandomPolicy::RandomPolicy()
  : Policy()
{
  random_engine = rosban_random::getRandomEngine();
}

Eigen::VectorXd RandomPolicy::getRawAction(const Eigen::VectorXd &state)
{
  return getRawAction(state, &random_engine);
}

Eigen::VectorXd RandomPolicy::getRawAction(const Eigen::VectorXd &state,
                                            std::default_random_engine * external_engine) const
{
  (void)state;
  bool delete_engine = false;
  if (external_engine == nullptr)
  {
    delete_engine = true;
    external_engine = rosban_random::newRandomEngine();
  }
  Eigen::VectorXd cmd(action_limits.rows());
  for (int dim = 0; dim < cmd.rows(); dim++)
  {
    std::uniform_real_distribution<double> distrib(action_limits(dim,0), action_limits(dim,1));
    cmd(dim) = distrib(*external_engine);
  }
  if (delete_engine) { delete(external_engine); }
  return cmd;
}

void RandomPolicy::to_xml(std::ostream & out) const
{
  (void)out;
}

void RandomPolicy::from_xml(TiXmlNode * node)
{
  (void)node;
}

std::string RandomPolicy::class_name() const
{
  return "random_policy";
}

}
