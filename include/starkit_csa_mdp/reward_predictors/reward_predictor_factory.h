#pragma once

#include "starkit_csa_mdp/reward_predictors/reward_predictor.h"

#include "starkit_utils/serialization/factory.h"

namespace csa_mdp
{
class RewardPredictorFactory : public starkit_utils::Factory<RewardPredictor>
{
public:
  RewardPredictorFactory();
};

}  // namespace csa_mdp
