#include "rosban_csa_mdp/solvers/model_based_learner.h"

#include "rosban_csa_mdp/core/fa_policy.h"

namespace csa_mdp
{

ModelBasedLearner::ModelBasedLearner()
  : value_steps(5), discount(0.98)
{
  //TODO remove code, it is only temporary. also remove associated headers
  //...
}

void ModelBasedLearner::internalUpdate()
{
  //TODO: update transition model and cost model
  updateValue();
  updatePolicy();
}

void ModelBasedLearner::updateValue()
{
  if (!reward_predictor)
  {
    throw std::logic_error("ModelBasedLearner::updateValue: reward predictor is not initialized");
  }
  if (!value_trainer)
  {
    throw std::logic_error("ModelBasedLearner::updateValue: value trainer is not initialized");
  }
  int nb_samples = samples.size();
  Eigen::MatrixXd inputs(model->stateDims(), nb_samples);
  Eigen::VectorXd observations(nb_samples);
  RewardPredictor::RewardFunction reward_function =
    [this] (const Eigen::VectorXd & state,
            const Eigen::VectorXd & action,
            const Eigen::VectorXd & next_state)
  {
    return this->model->getReward(state, action, next_state);
  };
  for (int sample = 0; sample < nb_samples; sample++)
  {
    Eigen::VectorXd state = samples[sample].state;
    double mean, var;
    reward_predictor->predict(state, policy, value_steps, model,
                              reward_function, discount,
                              &mean, &var);
    inputs.col(sample) = state;
    observations(sample) = mean;
  }
  value = value_trainer->train(inputs, observations, model->getStateLimits());
}

void ModelBasedLearner::updatePolicy()
{
  if (!action_optimizer)
  {
    throw std::logic_error("ModelBasedLearner::updateValue: action optimizer is not initialized");
  }
  if (!policy_trainer)
  {
    throw std::logic_error("ModelBasedLearner::updateValue: policy trainer is not initialized");
  }
  int nb_samples = samples.size();
  Eigen::MatrixXd inputs(model->stateDims(), nb_samples);
  Eigen::MatrixXd observations(nb_samples, model->actionDims());
  ActionOptimizer::RewardFunction reward_function =
    [this] (const Eigen::VectorXd & state,
            const Eigen::VectorXd & action,
            const Eigen::VectorXd & next_state)
  {
    return this->model->getReward(state, action, next_state);
  };
  ActionOptimizer::ValueFunction value_function =
    [this] (const Eigen::VectorXd & state)
  {
    double mean, var;
    this->value->predict(state, mean, var);
    return mean;
  };
  for (int sample = 0; sample < nb_samples; sample++)
  {
    Eigen::VectorXd state = samples[sample].state;
    Eigen::VectorXd best_action;
    best_action = action_optimizer->optimize(state, policy, model,
                                             reward_function, value_function,
                                             discount);
    inputs.col(sample) = state;
    observations.row(sample) = best_action.transpose();
  }
  std::unique_ptr<rosban_fa::FunctionApproximator> policy_fa;
  policy_fa = policy_trainer->train(inputs, observations, model->getStateLimits());
  policy = std::unique_ptr<Policy>(new FAPolicy(std::move(policy_fa)));
}

}
