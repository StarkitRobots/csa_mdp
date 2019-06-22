#include "starkit_csa_mdp/reward_predictors/reward_predictor.h"

namespace csa_mdp
{
RewardPredictor::RewardPredictor() : nb_threads(1)
{
}
RewardPredictor::~RewardPredictor()
{
}

void RewardPredictor::setNbThreads(int new_nb_threads)
{
  nb_threads = new_nb_threads;
}

Json::Value RewardPredictor::toJson() const
{
  Json::Value v;
  v["nb_threads"] = nb_threads;
  return v;
}

void RewardPredictor::fromJson(const Json::Value& v, const std::string& dir_name)
{
  (void)dir_name;
  starkit_utils::tryRead(v, "nb_threads", &nb_threads);
}

}  // namespace csa_mdp
