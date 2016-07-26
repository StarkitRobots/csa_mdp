#include "rosban_csa_mdp/solvers/model_based_learner.h"

#include "rosban_csa_mdp/action_optimizers/action_optimizer_factory.h"
#include "rosban_csa_mdp/core/fa_policy.h"
#include "rosban_csa_mdp/core/problem_factory.h"
#include "rosban_csa_mdp/core/random_policy.h"

#include "rosban_fa/trainer_factory.h"

#include "rosban_random/tools.h"

#include "rosban_utils/multi_core.h"

using csa_mdp::ProblemFactory;
using csa_mdp::RandomPolicy;
using rosban_fa::Trainer;
using rosban_fa::TrainerFactory;

using rosban_utils::MultiCore;
using rosban_utils::TimeStamp;

// Default types
#include "rosban_csa_mdp/reward_predictors/monte_carlo_predictor.h"
#include "rosban_csa_mdp/action_optimizers/basic_optimizer.h"
#include "rosban_fa/gp_forest_trainer.h"
#include "rosban_fa/pwl_forest_trainer.h"

using csa_mdp::MonteCarloPredictor;
using csa_mdp::BasicOptimizer;
using rosban_fa::GPForestTrainer;
using rosban_fa::PWLForestTrainer;

namespace csa_mdp
{

ModelBasedLearner::ModelBasedLearner()
  : reward_predictor(new MonteCarloPredictor()),
    value_trainer(new GPForestTrainer()),
    action_optimizer(new BasicOptimizer()),
    policy_trainer(new PWLForestTrainer()),
    value_steps(5)
{
  engine = rosban_random::getRandomEngine();
  //TODO add experimental code and remove it later (with associated headers)
}

const std::shared_ptr<const Policy> ModelBasedLearner::getPolicy() const
{
  if (policy) return policy;
  std::unique_ptr<Policy> fake_policy(new RandomPolicy);
  fake_policy->setActionLimits(getActionLimits());
  return std::move(fake_policy);
}

Eigen::VectorXd ModelBasedLearner::getAction(const Eigen::VectorXd & state)
{
  return getPolicy()->getAction(state, &engine);
}
bool ModelBasedLearner::hasAvailablePolicy()
{
  if (policy) return true;
  return false;
}

void ModelBasedLearner::savePolicy(const std::string & prefix)
{
  std::cerr << "Warning: ModelBasedLearner::savePolicy is not implemented" << std::endl;
}

void ModelBasedLearner::saveStatus(const std::string & prefix)
{
  std::cerr << "Warning: ModelBasedLearner::saveStatus is not implemented" << std::endl;
}


void ModelBasedLearner::internalUpdate()
{
  time_repartition.clear();
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
  Eigen::MatrixXd inputs(getStateLimits().rows(), nb_samples);
  Eigen::VectorXd observations(nb_samples);
  RewardPredictor::RewardFunction reward_function =
    [this] (const Eigen::VectorXd & state,
            const Eigen::VectorXd & action,
            const Eigen::VectorXd & next_state)
  {
    return this->model->getReward(state, action, next_state);
  };
  ActionOptimizer::ValueFunction value_function =
    [this] (const Eigen::VectorXd & state)
  {
    if (!this->value) return 0.0;
    double mean, var;
    this->value->predict(state, mean, var);
    return mean;
  };
  TimeStamp start_reward_predictor = TimeStamp::now();
  for (int sample = 0; sample < nb_samples; sample++)
  {
    Eigen::VectorXd state = samples[sample].state;
    double mean, var;
    reward_predictor->predict(state, getPolicy(), value_steps, model,
                              reward_function, value_function, discount,
                              &mean, &var);
    inputs.col(sample) = state;
    observations(sample) = mean;
  }
  TimeStamp end_reward_predictor = TimeStamp::now();
  value = value_trainer->train(inputs, observations, getStateLimits());
  TimeStamp end_value_trainer = TimeStamp::now();
  time_repartition["reward_predictor"] = diffSec(start_reward_predictor, end_reward_predictor);
  time_repartition["value_trainer"   ] = diffSec(end_reward_predictor  , end_value_trainer   );
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
  Eigen::MatrixXd inputs(getStateLimits().rows(), nb_samples);
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
    if (!this->value) return 0.0;
    double mean, var;
    this->value->predict(state, mean, var);
    return mean;
  };
  TimeStamp start_action_optimizer = TimeStamp::now();
  MultiCore::StochasticTask ao_task;
  ao_task = [this, reward_function, value_function, &inputs, &observations]
    (int start_idx, int end_idx, std::default_random_engine * thread_engine)
    {
      for (int sample = start_idx; sample < end_idx; sample++)
      {
        Eigen::VectorXd state = this->samples[sample].state;
        Eigen::VectorXd best_action;
        best_action = this->action_optimizer->optimize(state, this->getPolicy(), 
                                                       this->model,
                                                       reward_function,
                                                       value_function,
                                                       this->discount,
                                                       thread_engine);
        inputs.col(sample) = state;
        observations.row(sample) = best_action.transpose();
      }
    };
  // Choosing how many threads will be used for the samples and how many
  // subthreads will be spawned by each thread
  int wished_threads = nb_threads;
  int subthreads = 1;
  // Increasing performances when the number of samples is small, but ensuring that
  // no more than nb_threads are created
  if (nb_threads > nb_samples) {
    subthreads = std::ceil(nb_threads / (double)nb_samples);
    wished_threads = nb_threads / subthreads;
  }
  action_optimizer->setNbThreads(subthreads);

  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = rosban_random::getRandomEngines(wished_threads, &engine);
  // Run threads in parallel
  MultiCore::runParallelStochasticTask(ao_task, nb_samples, &engines);

  TimeStamp end_action_optimizer = TimeStamp::now();
  std::unique_ptr<rosban_fa::FunctionApproximator> new_policy_fa;
  new_policy_fa = policy_trainer->train(inputs, observations, getStateLimits());
  std::unique_ptr<Policy> new_policy(new FAPolicy(std::move(new_policy_fa)));
  new_policy->setActionLimits(getActionLimits());
  policy = std::move(new_policy);
  TimeStamp end_policy_trainer = TimeStamp::now();
  time_repartition["action_optimizer"] = diffSec(start_action_optimizer, end_action_optimizer);
  time_repartition["policy_trainer"  ] = diffSec(end_action_optimizer  , end_policy_trainer  );
}

std::string ModelBasedLearner::class_name() const
{ return "ModelBasedLearner"; }

void ModelBasedLearner::to_xml(std::ostream &out) const
{
  Learner::to_xml(out);
  if (model)            model->factoryWrite("model", out);
  if (reward_predictor) reward_predictor->factoryWrite("reward_predictor", out);
  if (value_trainer)    value_trainer->factoryWrite("value_trainer", out);
  if (action_optimizer) action_optimizer->factoryWrite("action_optimizer", out);
  if (policy_trainer)   policy_trainer->factoryWrite("policy_trainer", out);
  rosban_utils::xml_tools::write<int>   ("values_steps", value_steps, out);
}

void ModelBasedLearner::from_xml(TiXmlNode *node)
{
  Learner::from_xml(node);
  model  = ProblemFactory().read(node, "model");
  //TODO
  //RewardPredictorFactory().tryRead("reward_predictor", node, reward_predictor);
  TrainerFactory().tryRead(node, "value_trainer", value_trainer);
  ActionOptimizerFactory().tryRead(node, "action_optimizer", action_optimizer);
  TrainerFactory().tryRead(node, "policy_trainer", policy_trainer);
  rosban_utils::xml_tools::try_read<int>   (node, "values_steps", value_steps);
}

}
