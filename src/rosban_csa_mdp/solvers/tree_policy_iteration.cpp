#include "rosban_csa_mdp/solvers/tree_policy_iteration.h"

#include "rosban_csa_mdp/core/fa_policy.h"
#include "rosban_csa_mdp/core/policy_factory.h"
#include "rosban_csa_mdp/core/random_policy.h"
#include "rosban_csa_mdp/value_approximators/value_approximator_factory.h"

#include "rosban_fa/optimizer_trainer_factory.h"

using rosban_utils::TimeStamp;

namespace csa_mdp
{

TreePolicyIteration::TreePolicyIteration()
  : best_score(std::numeric_limits<double>::lowest()),
    memoryless_policy_trainer(true),
    use_value_approximator(true)
{
}

TreePolicyIteration::~TreePolicyIteration() {}

void TreePolicyIteration::init(std::default_random_engine * engine)
{
  // Initialize a random policy if no policy has been provided
  if (!policy) {
    policy = std::unique_ptr<Policy>(new RandomPolicy());
  }
  // Reset policy if necessary
  policy->init();
  policy->setActionLimits(problem->getActionLimits());
  best_score = evaluatePolicy(*policy, engine);
  writeScore(best_score);
}

void TreePolicyIteration::update(std::default_random_engine * engine) {
  // Update function approximatiors
  if (use_value_approximator) {
    TimeStamp value_start = TimeStamp::now();
    updateValue(engine);
    TimeStamp value_end = TimeStamp::now();
    writeTime("updateValue", diffSec(value_start, value_end));
  }
  // Always update policy
  TimeStamp policy_start = TimeStamp::now();
  std::unique_ptr<Policy> new_policy = updatePolicy(engine);
  TimeStamp policy_end = TimeStamp::now();
  writeTime("updatePolicy", diffSec(policy_start, policy_end));
  // Evaluate the policy expected score
  double score = evaluatePolicy(*new_policy, engine);
  TimeStamp evaluation_end = TimeStamp::now();
  writeTime("evaluation", diffSec(policy_end, evaluation_end));
  writeScore(score);
  // Save value used to build policy if enabled
  if (use_value_approximator) {
    std::ostringstream oss_v;
    oss_v << "value" << iterations << ".bin";
    value->save(oss_v.str());
  }
  // Saving policy used at this iteration
  const FAPolicy & fap = dynamic_cast<const FAPolicy &>(*new_policy);
  std::ostringstream oss_p;
  oss_p << "policy" << iterations << ".bin";
  fap.saveFA(oss_p.str());
  // Replace policy if it had a better score
  if (score > best_score) {
    best_score = score;
    policy = std::move(new_policy);
  }

}

void TreePolicyIteration::updateValue(std::default_random_engine * engine) {
  value = value_approximator->train(*policy,
                                    *problem,
                                    [this](const Eigen::VectorXd & state)
                                    {
                                      if (!this->value) return 0.0;
                                      double mean, var;
                                      this->value->predict(state, mean, var);
                                      return mean;
                                    },
                                    discount,
                                    engine);
}

std::unique_ptr<Policy>
TreePolicyIteration::updatePolicy(std::default_random_engine * engine) {
  rosban_fa::OptimizerTrainer::RewardFunction reward_func =
    [this]
    (const Eigen::VectorXd & parameters,
     const Eigen::VectorXd & actions,
     std::default_random_engine * engine)
    {
      // Computing first step reward and successor
      Eigen::VectorXd state = parameters;
      Eigen::VectorXd next_state = problem->getSuccessor(state, actions, engine);
      double reward = problem->getReward(state, actions, next_state);
      double value, value_var;
      // If a value approximator is used, only one step is required
      if (this->use_value_approximator) {
        this->value->predict(next_state, value, value_var);
        return reward + this->discount * value;
      }
      // Otherwise, do multiple steps
      state = next_state;
      double gain = this->discount;
      for (int step = 1; step < this->trial_length; step++) {
        // Computing step
        Eigen::VectorXd action = policy->getAction(state, engine);
        next_state = problem->getSuccessor(state, action, engine);
        double step_reward = problem->getReward(state, actions, next_state);
        // Accumulating reward
        reward += step_reward * gain;
        // Updating values
        state = next_state;
        gain *= discount;
        // Stop iterations if we reached a terminal state
        if (problem->isTerminal(state)) break;
      }
      return reward;
    };
  if (memoryless_policy_trainer) {
    policy_trainer->reset();
  }
  policy_trainer->setParametersLimits(problem->getStateLimits());
  policy_trainer->setActionsLimits(problem->getActionLimits());
  std::unique_ptr<rosban_fa::FunctionApproximator> fa;
  fa = policy_trainer->train(reward_func, engine);
  std::unique_ptr<Policy> new_policy(new FAPolicy(std::move(fa)));
  new_policy->setActionLimits(problem->getActionLimits());
  return std::move(new_policy);
}

void TreePolicyIteration::setNbThreads(int nb_threads_)
{
  BlackBoxLearner::setNbThreads(nb_threads_);
  if (value_approximator)
    value_approximator->setNbThreads(nb_threads);
  if (policy_trainer)
    policy_trainer->setNbThreads(nb_threads);
}

std::string TreePolicyIteration::class_name() const {
  return "TreePolicyIteration";
}

void TreePolicyIteration::to_xml(std::ostream &out) const
{
  //TODO
  (void) out;
  throw std::logic_error("TreePolicyIteration::to_xml: not implemented");
}

void TreePolicyIteration::from_xml(TiXmlNode *node)
{
  // Calling parent implementation
  BlackBoxLearner::from_xml(node);
  // Reading simple parameters
  rosban_utils::xml_tools::try_read<bool>  (node, "memoryless_policy_trainer", memoryless_policy_trainer);
  rosban_utils::xml_tools::try_read<bool>  (node, "use_value_approximator", use_value_approximator);
  // Read value approximator if necessary
  if (use_value_approximator) {
    value_approximator = ValueApproximatorFactory().read(node, "value_approximator");
  }
  // Policy trainer is required
  policy_trainer = rosban_fa::OptimizerTrainerFactory().read(node, "policy_trainer");
  // Initial policy might be provided (not necessary)
  PolicyFactory().tryRead(node, "policy", policy);
  // Update number of threads for all
  setNbThreads(nb_threads);
}

}
