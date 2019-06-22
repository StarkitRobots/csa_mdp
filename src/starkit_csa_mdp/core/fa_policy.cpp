#include "starkit_csa_mdp/core/fa_policy.h"

#include "starkit_fa/function_approximator_factory.h"

#include "starkit_random/multivariate_gaussian.h"
#include "starkit_random/tools.h"

using namespace starkit_utils;
using starkit_fa::FunctionApproximatorFactory;
using starkit_random::MultivariateGaussian;

namespace csa_mdp
{
FAPolicy::FAPolicy() : apply_noise(false)
{
  engine = starkit_random::getRandomEngine();
}

FAPolicy::FAPolicy(std::unique_ptr<starkit_fa::FunctionApproximator> fa_) : fa(std::move(fa_)), apply_noise(false)
{
  engine = starkit_random::getRandomEngine();
}

void FAPolicy::setRandomness(bool new_apply_noise)
{
  apply_noise = new_apply_noise;
}

Eigen::VectorXd FAPolicy::getRawAction(const Eigen::VectorXd& state)
{
  return getRawAction(state, &engine);
}

Eigen::VectorXd FAPolicy::getRawAction(const Eigen::VectorXd& state, std::default_random_engine* external_engine) const
{
  bool delete_engine = false;
  if (apply_noise && external_engine == nullptr)
  {
    delete_engine = true;
    external_engine = starkit_random::newRandomEngine();
  }
  Eigen::VectorXd mean;
  Eigen::MatrixXd covar;
  fa->predict(state, mean, covar);

  Eigen::VectorXd cmd;
  if (apply_noise)
  {
    cmd = MultivariateGaussian(mean, covar).getSample(external_engine);
  }
  else
  {
    cmd = mean;
  }

  if (delete_engine)
  {
    delete (external_engine);
  }
  return cmd;
}

std::string FAPolicy::getClassName() const
{
  return "FAPolicy";
}

Json::Value FAPolicy::toJson() const
{
  Json::Value v;
  v["noise"] = apply_noise;
  throw std::logic_error("FAPolicy:toJson: not implemented");
}

void FAPolicy::fromJson(const Json::Value& v, const std::string& dir_name)
{
  std::string abs_path, rel_path, path;
  starkit_utils::tryRead<std::string>(v, "abs path", &abs_path);
  starkit_utils::tryRead<std::string>(v, "rel path", &rel_path);
  if (abs_path != "" && rel_path != "")
  {
    throw JsonParsingError("FAPolicy::fromJson: both 'abs path' and 'rel path' specified");
  }
  else if (abs_path == "" && rel_path == "")
  {
    throw JsonParsingError("FAPolicy::fromJson: no 'abs path' neither 'rel path' specified");
  }
  else if (abs_path != "")
  {
    path = abs_path;
  }
  else
  {
    path = dir_name + rel_path;
  }
  FunctionApproximatorFactory().loadFromFile(path, fa);
  starkit_utils::tryRead(v, "noise", &apply_noise);
}

void FAPolicy::saveFA(const std::string& filename) const
{
  if (!fa)
    return;
  fa->save(filename);
}

}  // namespace csa_mdp
