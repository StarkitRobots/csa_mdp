#include "rosban_csa_mdp/solvers/black_box_learner.h"

#include "rosban_csa_mdp/core/fa_policy.h"
#include "rosban_csa_mdp/core/policy_factory.h"
#include "rosban_csa_mdp/core/problem_factory.h"
#include "rosban_csa_mdp/core/random_policy.h"
#include "rosban_csa_mdp/value_approximators/value_approximator_factory.h"

#include "rosban_fa/optimizer_trainer_factory.h"

using rosban_utils::TimeStamp;

namespace csa_mdp
{

BlackBoxLearner::BlackBoxLearner()
  : nb_threads(1),
    time_budget(60),
    discount(0.98),
    trial_length(50),
    nb_evaluation_trials(100),
    best_score(std::numeric_limits<double>::lowest()),
    iterations(0)
{
  openLogs();
}

BlackBoxLearner::~BlackBoxLearner()
{
  closeLogs();
}

void BlackBoxLearner::run(std::default_random_engine * engine)
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
  // Main learning loop
  learning_start = rosban_utils::TimeStamp::now();
  while (true) {
    iterations++;
    std::cout << "Iteration " << iterations << std::endl;
    // Update function approximatiors
    TimeStamp value_start = TimeStamp::now();
    updateValue(engine);
    TimeStamp value_end = TimeStamp::now();
    writeTime("updateValue", diffSec(value_start, value_end));
    std::unique_ptr<Policy> new_policy = updatePolicy(engine);
    TimeStamp policy_end = TimeStamp::now();
    writeTime("updatePolicy", diffSec(value_end, policy_end));
    // Stop if time has elapsed
    double elapsed = diffSec(learning_start, rosban_utils::TimeStamp::now());
    if (elapsed > time_budget)
      break;
    // evaluate and save policy if it's better than previous ones
    double score = evaluatePolicy(*new_policy, engine);
    TimeStamp evaluation_end = TimeStamp::now();
    writeTime("evaluation", diffSec(policy_end, evaluation_end));
    writeScore(score);
    // Saving value and policy used at this iteration
    const FAPolicy & fap = dynamic_cast<const FAPolicy &>(*new_policy);
    std::ostringstream oss_p, oss_v;
    oss_p << "policy" << iterations << ".bin";
    oss_v << "value" << iterations << ".bin";
    fap.saveFA(oss_p.str());
    value->save(oss_v.str());
    // Replace policy if it had a better score
    if (score > best_score) {
      best_score = score;
      policy = std::move(new_policy);
    }
  }
}

void BlackBoxLearner::updateValue(std::default_random_engine * engine) {
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

std::unique_ptr<Policy> BlackBoxLearner::updatePolicy(std::default_random_engine * engine) {
  rosban_fa::OptimizerTrainer::RewardFunction reward_func =
    [this]
    (const Eigen::VectorXd & parameters,
     const Eigen::VectorXd & actions,
     std::default_random_engine * engine)
    {
      const Eigen::VectorXd & state = parameters;
      const Eigen::VectorXd & next_state = problem->getSuccessor(state, actions, engine);
      double reward = problem->getReward(state, actions, next_state);
      double value, value_var;
      this->value->predict(next_state, value, value_var);
      return reward + this->discount * value;
    };
  policy_trainer->setParametersLimits(problem->getStateLimits());
  policy_trainer->setActionsLimits(problem->getActionLimits());
  std::unique_ptr<rosban_fa::FunctionApproximator> fa;
  fa = policy_trainer->train(reward_func, engine);
  std::unique_ptr<Policy> new_policy(new FAPolicy(std::move(fa)));
  new_policy->setActionLimits(problem->getActionLimits());
  return std::move(new_policy);

}

double BlackBoxLearner::evaluatePolicy(const Policy & p,
                                       std::default_random_engine * engine)
{
  double reward = 0;
  for (int trial = 0; trial < nb_evaluation_trials; trial++) {
    Eigen::VectorXd state = problem->getStartingState(engine);
    double gain = 1.0;
    for (int step = 0; step < trial_length; step++) {
      Eigen::VectorXd action = p.getAction(state, engine);
      Eigen::VectorXd next_state = problem->getSuccessor(state, action, engine);
      double step_reward = problem->getReward(state, action, next_state);
      state = next_state;
      reward += gain * step_reward;
      gain = gain * discount;
      if (problem->isTerminal(state)) break;
    }
  }
  return reward / nb_evaluation_trials;
}

void BlackBoxLearner::setNbThreads(int nb_threads_)
{
  nb_threads = nb_threads_;
  if (value_approximator)
    value_approximator->setNbThreads(nb_threads);
  if (policy_trainer)
    policy_trainer->setNbThreads(nb_threads);
}

std::string BlackBoxLearner::class_name() const {
  return "BlackBoxLearner";
}

void BlackBoxLearner::to_xml(std::ostream &out) const
{
  //TODO
  (void) out;
  throw std::logic_error("BlackBoxLearner::to_xml: not implemented");
}

void BlackBoxLearner::from_xml(TiXmlNode *node)
{
  // Reading simple parameters
  rosban_utils::xml_tools::try_read<int>(node, "nb_threads", nb_threads);
  rosban_utils::xml_tools::try_read<int>(node, "trial_length", trial_length);
  rosban_utils::xml_tools::try_read<int>(node, "nb_evaluation_trials", nb_evaluation_trials);
  rosban_utils::xml_tools::try_read<double>(node, "time_budget", time_budget);
  rosban_utils::xml_tools::try_read<double>(node, "discount", discount);
  // Getting problem
  std::shared_ptr<const Problem> tmp_problem;
  tmp_problem = ProblemFactory().read(node, "problem");
  problem = std::dynamic_pointer_cast<const BlackBoxProblem>(tmp_problem);
  if (!problem) {
    throw std::runtime_error("BlackBoxLearner::from_xml: problem is not a BlackBoxProblem");
  }
  // Value approximator
  value_approximator = ValueApproximatorFactory().read(node, "value_approximator");
  policy_trainer = rosban_fa::OptimizerTrainerFactory().read(node, "policy_trainer");
  // Initial policy might be provided
  PolicyFactory().tryRead(node, "policy", policy);
  // Update number of threads for all
  setNbThreads(nb_threads);
}

void BlackBoxLearner::openLogs()
{
  // Opening files
  time_file.open("time.csv");
  results_file.open("results.csv");
  // Writing headers
  time_file << "iteration,part,time" << std::endl;
  results_file << "iteration,score" << std::endl;
}


void BlackBoxLearner::closeLogs()
{
  time_file.close();
  results_file.close();
}

void BlackBoxLearner::writeTime(const std::string & name, double time)
{
  //TODO:
  // - Add members once available (require implementation in AdaptativeTree)
  time_file << iterations << "," << name << "," << time << std::endl;
}

void BlackBoxLearner::writeScore(double score) {
  results_file << iterations << "," << score << std::endl;
}


}
