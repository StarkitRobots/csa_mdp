#include "rosban_csa_mdp/value_approximators/value_approximator.h"

#include "rosban_csa_mdp/reward_predictors/reward_predictor.h"

#include "rosban_fa/trainer.h"

namespace csa_mdp
{

class ExtraTreesApproximator : public ValueApproximator {
public:
  ExtraTreesApproximator();
  ~ExtraTreesApproximator();

  virtual std::unique_ptr<rosban_fa::FunctionApproximator>
  train(const Policy & policy,
        const Problem & problem,
        Problem::ValueFunction current_value_function,
        double discount,
        std::default_random_engine * engine) override;

  virtual void setNbThreads(int nb_threads) override;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

protected:
  /// Number of samples used as basis of the evaluation function
  int nb_samples;

  /// The predictor used to sample rewards
  std::unique_ptr<RewardPredictor> predictor;

  /// The trainer used to approximate the reward function
  std::unique_ptr<rosban_fa::Trainer> trainer;
};

}
