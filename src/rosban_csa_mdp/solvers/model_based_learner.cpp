#include "rosban_csa_mdp/solvers/model_based_learner.h"

#include "rosban_csa_mdp/action_optimizers/action_optimizer_factory.h"
#include "rosban_csa_mdp/core/fa_policy.h"
#include "rosban_csa_mdp/core/problem_factory.h"
#include "rosban_csa_mdp/core/random_policy.h"

#include "rosban_fa/trainer_factory.h"

#include "rosban_random/tools.h"

using csa_mdp::ProblemFactory;
using csa_mdp::RandomPolicy;
using rosban_fa::Trainer;
using rosban_fa::TrainerFactory;

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
  for (int sample = 0; sample < nb_samples; sample++)
  {
    Eigen::VectorXd state = samples[sample].state;
    Eigen::VectorXd best_action;
    best_action = action_optimizer->optimize(state, getPolicy(), model,
                                             reward_function, value_function,
                                             discount);
    inputs.col(sample) = state;
    observations.row(sample) = best_action.transpose();
  }
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
  if (model) {
    out << "<model>";
    model->write(model->class_name(), out);
    out << "</model>";
  }
  if (reward_predictor) {
    out << "<reward_predictor>";
    reward_predictor->write(reward_predictor->class_name(), out);
    out << "</reward_predictor>";
  }
  if (value_trainer) {
    out << "<value_trainer>";
    value_trainer->write(value_trainer->class_name(), out);
    out << "</value_trainer>";
  }
  if (action_optimizer) {
    out << "<action_optimizer>";
    action_optimizer->write(action_optimizer->class_name(), out);
    out << "</action_optimizer>";
  }
  if (policy_trainer) {
    out << "<policy_trainer>";
    policy_trainer->write(policy_trainer->class_name(), out);
    out << "</policy_trainer>";
  }
  rosban_utils::xml_tools::write<int>   ("values_steps", value_steps, out);
}

void ModelBasedLearner::from_xml(TiXmlNode *node)
{
  Learner::from_xml(node);
  // 1: read model (mandatory)
  TiXmlNode * model_node = node->FirstChild("model");
  if(!model_node) {
    throw std::runtime_error("Failed to find node 'model' in '" + node->ValueStr() + "'");
  }
  model  = std::shared_ptr<Problem>(ProblemFactory().build(model_node));
  // 2: read reward predictor (optional)
  TiXmlNode * reward_predictor_node = node->FirstChild("reward_predictor");
  if(reward_predictor_node) {
    //TODO
    //reward_predictor = std::unique_ptr<RewardPredictor>(RewardPredictorFactory().build(reward_predictor_node));
  }
  // 3: read value_trainer (optional)
  TiXmlNode * value_trainer_node = node->FirstChild("value_trainer");
  if(value_trainer_node) {
    value_trainer = std::unique_ptr<Trainer>(TrainerFactory().build(value_trainer_node));
  }
  // 4: read action_optimizer (optional)
  ActionOptimizerFactory().tryRead(node, "action_optimizer", action_optimizer);
  // 5: read policy_trainer (optional)
  TiXmlNode * policy_trainer_node = node->FirstChild("policy_trainer");
  if(policy_trainer_node) {
    policy_trainer = std::unique_ptr<Trainer>(TrainerFactory().build(policy_trainer_node));
  }
  rosban_utils::xml_tools::try_read<int>   (node, "values_steps", value_steps);
}

}
