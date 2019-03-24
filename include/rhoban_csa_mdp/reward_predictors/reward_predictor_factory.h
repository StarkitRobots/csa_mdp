#pragma once

#include "rhoban_csa_mdp/reward_predictors/reward_predictor.h"

#include "rhoban_utils/serialization/factory.h"

namespace csa_mdp
{
class RewardPredictorFactory : public rhoban_utils::Factory<RewardPredictor>
{
public:
  RewardPredictorFactory();
};

}  // namespace csa_mdp
