#include "rosban_csa_mdp/core/fa_policy.h"

#include "rosban_random/multivariate_gaussian.h"
#include "rosban_random/tools.h"

using rosban_random::MultiVariateGaussian;

namespace csa_mdp
{

FAPolicy::FAPolicy(std::unique_ptr<rosban_fa::FunctionApproximator> fa_)
  : fa(std::move(fa_)), apply_noise(false)
{
  engine = rosban_random::getRandomEngine();
}

void FAPolicy::setRandomness(bool new_apply_noise)
{
  apply_noise = new_apply_noise;
}

Eigen::VectorXd FAPolicy::getRawAction(const Eigen::VectorXd &state)
{
  return getRawAction(state, &engine);
}

Eigen::VectorXd FAPolicy::getRawAction(const Eigen::VectorXd &state,
                                       std::default_random_engine * external_engine) const
{
  bool delete_engine = false;
  if (apply_noise && external_engine == nullptr)
  {
    delete_engine = true;
    external_engine = rosban_random::newRandomEngine();
  }
  Eigen::VectorXd mean;
  Eigen::MatrixXd covar;
  fa->predict(state, mean, covar);

  Eigen::VectorXd cmd;
  if (apply_noise) {
    cmd = MultiVariateGaussian(mean, covar).getSample(*external_engine);
  }
  else {
    cmd = mean;
  }

  if (delete_engine) { delete(external_engine); }
  return cmd;
}

std::string FAPolicy::class_name() const
{
  return "FAPolicy";
}

void FAPolicy::to_xml(std::ostream & out) const
{
  rosban_utils::xml_tools::write<bool>("noise", apply_noise, out);
}

void FAPolicy::from_xml(TiXmlNode * node)
{
  rosban_utils::xml_tools::try_read<bool>(node, "noise", apply_noise);
}

void FAPolicy::saveFA(const std::string & filename) const
{
  if (!fa) return;
  fa->save(filename);
}

}
