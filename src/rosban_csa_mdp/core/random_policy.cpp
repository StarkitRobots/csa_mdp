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
  if (external_engine == nullptr) {
    delete_engine = true;
    external_engine = rosban_random::newRandomEngine();
  }
  // Choosing action_id randomly
  std::uniform_int_distribution<int> action_distrib(0, action_limits.size() - 1);
  int action_id = action_distrib(*external_engine);
  // Choosing randomly among continuous dimensions
  const Eigen::MatrixXd & limits = action_limits[action_id];
  Eigen::VectorXd raw_action(limits.rows() + 1);
  raw_action(0) = action_id;
  for (int dim = 0; dim < limits.rows(); dim++) {
    std::uniform_real_distribution<double> distrib(limits(dim,0), limits(dim,1));
    raw_action(dim + 1) = distrib(*external_engine);
  }
  if (delete_engine) { delete(external_engine); }
  return raw_action;
}

Json::Value RandomPolicy::toJson() const
{
  return Json::Value();
}

void RandomPolicy::fromJson(const Json::Value & v, const std::string & dir_name)
{
  (void)v;
  (void)dir_name;
}

std::string RandomPolicy::getClassName() const
{
  return "random_policy";
}

}
