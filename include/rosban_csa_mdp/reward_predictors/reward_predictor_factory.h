#pragma once

#include "rosban_csa_mdp/reward_predictors/reward_predictor.h"

#include "rosban_utils/factory.h"

namespace csa_mdp
{

class RewardPredictorFactory : public rosban_utils::Factory<RewardPredictor>
{
public:
  RewardPredictorFactory();
};

}
