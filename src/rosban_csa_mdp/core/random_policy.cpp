#include "rosban_csa_mdp/core/random_policy.h"

#include "rosban_regression_forests/tools/random.h"

namespace csa_mdp
{

RandomPolicy::RandomPolicy()
  : Policy()
{
  random_engine = regression_forests::get_random_engine();
}

Eigen::VectorXd RandomPolicy::getRawAction(const Eigen::VectorXd &state)
{
  (void)state;
  Eigen::VectorXd cmd(action_limits.rows());
  for (int dim = 0; dim < cmd.rows(); dim++)
  {
    std::uniform_real_distribution<double> distrib(action_limits(dim,0), action_limits(dim,1));
    cmd(dim) = distrib(random_engine);
  }
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
