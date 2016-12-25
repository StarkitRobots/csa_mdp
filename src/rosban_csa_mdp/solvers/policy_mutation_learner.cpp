#include "rosban_csa_mdp/solvers/policy_mutation_learner.h"

#include "rosban_bbo/optimizer_factory.h"

namespace csa_mdp
{

PolicyMutationLearner::PolicyMutationLearner()
{
}

PolicyMutationLearner::~PolicyMutationLearner() {}

void PolicyMutationLearner::init(std::default_random_engine * engine) {
  const Eigen::MatrixXd & action_limits = problem->getActionLimits();
  if (policy) {
    // TODO:
    // evaluate and print score from expert policy
    // Approximate policy
    throw std::logic_error("PolicyMutationLearner::init: import from policy not implemented");
  }
  else {
    std::unique_ptr<Split> split(new FakeSplit());
    // Default action is the middle of the space
    Eigen::VectorXd action = (action_limits.col(0) + action_limits.col(1)) / 2;
    std::unique_ptr<FunctionApproximator> default_fa(new ConstantApproximator(action));
    policy_tree = FATree(std::move(split), {std::move(action)});
    policy = buildPolicy(policy_tree);
    // Adding mutation candidate
    MutationCandidate candidate;
    candidate.space = problem->getStateLimits();
    candidate.post_training_score = localEvaluation(*policy,
                                                    candidate.space,
                                                    training_evaluations,
                                                    engine);
    candidate.mutation_score = 1.0;
    candidate.last_training = 0;
    candidate.is_leaf = true;
    mutation_candidates.push_back(candidate);
  }
  double avg_reward = evaluatePolicy(*policy, engine);
  std::cout << "Initial Reward: " << avg_reward << std::endl;
}

void PolicyMutationLearner::update(std::default_random_engine * engine) {
  int mutation_id = getMutationId(engine);
  mutate(mutation_id, engine);
  double new_reward = evaluatePolicy(*policy, engine);
  std::cout << "New reward: " << new_reward << std::endl;
  // TODO update weights
}

int PolicyMutationLearner::getMutationId(std::default_random_engine * engine) {
  // Getting max_score
  double total_score = 0;
  for (const MutationCandidate & c : mutation_candidates) {
    total_score += c.mutation_score;
  }
  // Getting corresponding element
  double c_score = std::uniform_real_distribution<double>(0, total_score)(*engine);

  double acc = 0;
  for (size_t id = 0, id < mutation_candidates.size(); id++) {
    acc += mutation_candidates[id].mutation_score;
    if (acc > c_score) {
      return id;
    }
  }
  // Should never happen except with numerical errors
  std::ostringstream oss;
  oss << "PolicyMutationLearner::getMutationId: c_score >= total_score"
      << " (" << c_score << ">=" << total_score << ")";
  throw std::logic_error(oss.str());
}

void PolicyMutationLearner::setNbThreads(int nb_threads) {
  BlackBoxLearner::setNbThreads(nb_threads_);
  optimizer->setNbThreads(nb_threads);
}

void PolicyMutationLearner::mutate(int mutation_id,
                                   std::default_random_engine * engine) {
  if (mutation_candidates[mutation_id].is_leaf) {
    mutateLeaf(mutation_id, engine);
  }
  else {
    mutatePreLeaf(mutation_id, engine);
  }
}

std::string PolicyMutationLearner::class_name() const {
  return "PolicyMutationLearner";
}

void PolicyMutationLearner::mutateLeaf(int mutation_id,
                                       std::default_random_engine * engine) {
  //TODO random choice between split and refineMutation
  refineMutation(mutation_id, engine);
}

void PolicyMutationLearner::mutatePreLeaf(int mutation_id,
                                          std::default_random_engine * engine) {
  //TODO
}

void PolicyMutationLearner::refineMutation(int mutation_id,
                                           std::default_random_engine * engine) {
  std::cout << "-> Applying a refine mutation" << std::endl;
  // Get reference to the appropriate mutation
  MutationCandidate * mutation = &(mutation_candidates[mutation_id]);
  // Get space and center
  Eigen::MatrixXd space = mutation->space;
  int input_dim = problem->stateDims();
  int output_dim = problem->actionDims();
  // Training function
  // TODO: use other models than PWL
  // TODO: something global should be done for guesses and models
  rosban_bbo::Optimizer::RewardFunc reward_func =
    [this, space, input_dim, output_dim]
    (const Eigen::VectorXd & parameters,
     std::default_random_engine * engine)
    {
      std::unique_ptr<FunctionApproximator> new_approximator(
        new LinearApproximator(input_dim, output_dim, parameters));
      std::unique_ptr<FATree> new_tree;
      new_tree = policy_tree->copyAndReplaceLeaf(space_center, std::move(new_approximator));
      FAPolicy policy(new_tree);
      return localEvaluation(*policy, space, training_evaluations, engine);
    };
  // Getting parameters_space
  bool narrow_slope = false;
  // TODO: options for narrow_slope probability
  Eigen::MatrixXd parameters_space;
  parameters_space = LinearApproximator::getParametersSpace(parameters_limits,
                                                            action_limits,
                                                            narrow_slope);
  // Computing initial parameters
  // TODO: using guesses
  Eigen::VectorXd initial_parameters;
  initial_params = LinearApproximator::getDefaultParameters(parameters_limits,
                                                            action_limits);
  optimizer->setLimits(parameters_space);
  // Creating refined approximator
  Eigen::VectorXd refined_parameters = optimizer->train(initial_parameters);
  std::unique_ptr<FunctionApproximator> refined_approximator(
        new LinearApproximator(input_dim, output_dim, parameters));
  // Creating refined policy
  std::unique_ptr<FATree> refined_tree;
  refined_tree = policy_tree->copyAndReplaceLeaf(space_center, std::move(refined_approximator));
  std::unique_ptr<Policy> new_policy = buildPolicy(*refined_tree);
  // Evaluate initial and refined policy with updated parameters (on local space)
  double initial_reward = localEvaluation(*policy, space, nb_evaluation_trials, engine);
  double refined_reward = localEvaluation(*refined_policy, space, nb_evaluation_trials, engine);
  std::cout << "\tinitial reward: " << initial_reward << std::endl;
  std::cout << "\trefined reward: " << refined_reward << std::endl;
  // Replace current if improvement has been seen
  if (refined_reward > initial_reward) {
    policy_tree = std::move(refined_tree);
    policy = std::move(refined_policy);
    mutation->post_training_score = refined_reward;
  }
  // Update mutation properties:
  mutation->last_training = iterations;
}

void PolicyMutationLearner::to_xml(std::ostream &out) const {
  //TODO
  (void) out;
  throw std::logic_error("PolicyMutationLearner::to_xml: not implemented");
}

void PolicyMutationLearner::from_xml(TiXmlNode *node) {
  // Calling parent implementation
  BlackBoxLearner::from_xml(node);
  // Reading class variables
  rosban_utils::xml_tools::try_read<int>(node, "training_evaluations", training_evaluations);
  // Optimizer is mandatory
  optimizer = rosban_bbo::OptimizerFactory().read(node, "optimizer");
  // Read Policy if provided (optional)
  PolicyFactory().tryRead(node, "policy", policy);
  // Synchronize number of threads
  setNbThreads(nb_threads);
}

std::unique_ptr<Policy> PolicyMutationLearner::buildPolicy(const FATree & tree) {
  std::unique_ptr<FATree> tree_copy(tree.clone());
  return std::unique_ptr<Policy>(new FAPolicy(std::move(tree_copy)));
}

}
