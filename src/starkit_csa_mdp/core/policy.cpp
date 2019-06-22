#include "starkit_csa_mdp/core/policy.h"

namespace csa_mdp
{
Policy::Policy() : internal_random_engine(std::random_device()()), nb_threads(1)
{
  init();
}

void Policy::init()
{
}

void Policy::setActionLimits(const std::vector<Eigen::MatrixXd>& limits)
{
  action_limits = limits;
}

Eigen::VectorXd Policy::boundAction(const Eigen::VectorXd& raw_action) const
{
  int action_id = (int)raw_action(0);
  int max_action = action_limits.size() - 1;
  if (action_id < 0 || action_id > max_action)
  {
    std::ostringstream oss;
    oss << "Policy::boundAction: action_id is invalid: " << action_id << " is not in [0," << max_action << "]";
    throw std::runtime_error(oss.str());
  }
  const Eigen::MatrixXd& limits = action_limits[action_id];
  int nb_action_dims = raw_action.rows() - 1;
  if (nb_action_dims != limits.rows())
  {
    std::ostringstream oss;
    oss << "Policy::boundAction: Number of rows does not match (" << nb_action_dims << " while " << limits.rows()
        << " were expected)";
    throw std::runtime_error(oss.str());
  }
  Eigen::VectorXd action = raw_action;
  for (int dim = 1; dim < raw_action.rows(); dim++)
  {
    action(dim) = std::min(limits(dim - 1, 1), std::max(limits(dim - 1, 0), action(dim)));
  }
  return action;
}

Eigen::VectorXd Policy::getAction(const Eigen::VectorXd& state)
{
  return boundAction(getRawAction(state));
}

Eigen::VectorXd Policy::getAction(const Eigen::VectorXd& state, std::default_random_engine* external_engine) const
{
  return boundAction(getRawAction(state, external_engine));
}

Eigen::VectorXd Policy::getRawAction(const Eigen::VectorXd& state)
{
  return getRawAction(state, &internal_random_engine);
}

std::unique_ptr<starkit_fa::FATree> Policy::extractFATree() const
{
  // TODO: approximate current policy (raw actions) by a FATree
  throw std::logic_error("Policy::extractFATree: unimplemented");
}

Json::Value Policy::toJson() const
{
  Json::Value v;
  v["nb_threads"] = nb_threads;
  return v;
}

void Policy::fromJson(const Json::Value& v, const std::string& dir_name)
{
  (void)dir_name;
  starkit_utils::tryRead(v, "nb_threads", &nb_threads);
}

void Policy::setNbThreads(int new_nb_threads)
{
  nb_threads = new_nb_threads;
}

}  // namespace csa_mdp
