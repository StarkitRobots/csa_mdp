#include "rosban_csa_mdp/solvers/model_based_learner.h"

namespace csa_mdp
{

void ModelBasedLearner::internalUpdate()
{
  // 1. update transition model
  // TODO
  // 2. Update value Model
  updateValue();
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
  Eigen::MatrixXd inputs(problem.stateDims(), nb_samples);
  Eigen::VectorXd observations(nb_samples);
  for (int sample = 0; sample < nb_samples; sample++)
  {
    Eigen::VectorXd state = samples[samples].state;
    double mean, double var;
    reward_predictor->predict(state, policy, value_steps, model, discount,
                              &mean, &var);
    inputs.col(sample) = state;
    observations(sample) = mean;
  }
  value = value_trainer->train(inputs, observations, problem->getStateLimits());
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
  Eigen::MatrixXd inputs(problem.stateDims(), nb_samples);
  Eigen::MatrixXd observations(nb_samples, problem.actionDims());
  for (int sample = 0; sample < nb_samples; sample++)
  {
    Eigen::VectorXd state = samples[samples].state;
    Eigen::VectorXd best_action;
    best_action = action_optimizer->optimize(...);
    inputs.col(sample) = state;
    observations.row(sample) = best_action.transpose();
  }
  policy = policy_trainer->train(inputs, observations, problem->getStateLimits());
}

}
