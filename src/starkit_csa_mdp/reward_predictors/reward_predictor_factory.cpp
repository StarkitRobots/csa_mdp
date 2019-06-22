#include "starkit_csa_mdp/reward_predictors/reward_predictor_factory.h"
#include "starkit_csa_mdp/reward_predictors/monte_carlo_predictor.h"

namespace csa_mdp
{
RewardPredictorFactory::RewardPredictorFactory()
{
  registerBuilder("MonteCarloPredictor", []() { return std::unique_ptr<RewardPredictor>(new MonteCarloPredictor); });
}

}  // namespace csa_mdp
