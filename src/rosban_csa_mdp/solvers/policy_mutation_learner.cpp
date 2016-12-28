#include "rosban_csa_mdp/solvers/policy_mutation_learner.h"

#include "rosban_bbo/optimizer_factory.h"

#include "rosban_csa_mdp/core/fa_policy.h"
#include "rosban_csa_mdp/core/policy_factory.h"

#include "rosban_fa/constant_approximator.h"
#include "rosban_fa/fake_split.h"
#include "rosban_fa/function_approximator_factory.h"
#include "rosban_fa/linear_approximator.h"
#include "rosban_fa/orthogonal_split.h"

using namespace rosban_fa;

namespace csa_mdp
{

PolicyMutationLearner::PolicyMutationLearner()
  : training_evaluations(50),
    split_probability(0.1)
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
    std::vector<std::unique_ptr<FunctionApproximator>> fas;
    fas.push_back(std::move(default_fa));
    policy_tree.reset(new FATree(std::move(split), fas));
    policy = buildPolicy(*policy_tree);
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
  for (size_t id = 0; id < mutation_candidates.size(); id++) {
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
  BlackBoxLearner::setNbThreads(nb_threads);
  //optimizer->setNbThreads(nb_threads);
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
  //TODO introduce two different types of mutations (refining + renewing)
  double rand_val = std::uniform_real_distribution<double>(0.0,1.0)(*engine);
  if (rand_val < split_probability) {
    splitMutation(mutation_id, engine);
  }
  else {
    refineMutation(mutation_id, engine);
  }
}

void PolicyMutationLearner::mutatePreLeaf(int mutation_id,
                                          std::default_random_engine * engine) {
  (void) mutation_id;
  (void) engine;
  //TODO
}

void PolicyMutationLearner::refineMutation(int mutation_id,
                                           std::default_random_engine * engine) {
  // Get reference to the appropriate mutation
  MutationCandidate * mutation = &(mutation_candidates[mutation_id]);
  // Get space and center
  Eigen::MatrixXd space = mutation->space;
  Eigen::MatrixXd action_limits = problem->getActionLimits();
  int input_dim = problem->stateDims();
  int output_dim = problem->actionDims();
  Eigen::VectorXd space_center = (space.col(0) + space.col(1)) / 2;
  // Debug:
  std::cout << "-> Applying a refine mutation on space" << std::endl
            << space.transpose() << std::endl;
  // Training function
  // TODO: use other models than PWL
  // TODO: something global should be done for guesses and models
  rosban_bbo::Optimizer::RewardFunc reward_func =
    [this, space, input_dim, output_dim, space_center]
    (const Eigen::VectorXd & parameters,
     std::default_random_engine * engine)
    {
      std::unique_ptr<FunctionApproximator> new_approximator(
        new LinearApproximator(input_dim, output_dim, parameters, space_center));
      std::unique_ptr<FATree> new_tree;
      new_tree = policy_tree->copyAndReplaceLeaf(space_center, std::move(new_approximator));
      //TODO: avoid this second copy of the tree which is not necessary
      std::unique_ptr<Policy> policy = buildPolicy(*new_tree);
      return localEvaluation(*policy, space, training_evaluations, engine);
    };
  // Getting parameters_space
  bool narrow_slope = false;
  // TODO: options for narrow_slope probability
  Eigen::MatrixXd parameters_space;
  parameters_space = LinearApproximator::getParametersSpace(space,
                                                            action_limits,
                                                            narrow_slope);
  // Computing initial parameters
  // TODO: using guesses
  Eigen::VectorXd initial_parameters;
  initial_parameters = LinearApproximator::getDefaultParameters(space,
                                                                action_limits);
  optimizer->setLimits(parameters_space);
  // Creating refined approximator
  Eigen::VectorXd refined_parameters = optimizer->train(reward_func,
                                                        initial_parameters,
                                                        engine);
  std::cout << "\tRefined parameters:" << std::endl
            << "\t\t" << refined_parameters.transpose() << std::endl;

  std::unique_ptr<FunctionApproximator> refined_approximator(
    new LinearApproximator(input_dim, output_dim, refined_parameters, space_center));
  // Creating refined policy
  std::unique_ptr<FATree> refined_tree;
  refined_tree = policy_tree->copyAndReplaceLeaf(space_center, std::move(refined_approximator));
  std::unique_ptr<Policy> refined_policy = buildPolicy(*refined_tree);
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

void PolicyMutationLearner::splitMutation(int mutation_id,
                                          std::default_random_engine * engine) {
  std::cout << "-> Applying a split mutation" << std::endl;
  const MutationCandidate & mutation = mutation_candidates[mutation_id];
  Eigen::MatrixXd space = mutation.space;
  Eigen::VectorXd space_center = (space.col(0) + space.col(1)) / 2;
  // Testing all dimensions as split and keeping the best one
  double best_score = std::numeric_limits<double>::lowest();
  std::unique_ptr<FATree> best_tree;
  for (int dim = 0; dim < problem->stateDims(); dim++) {
    double score;
    std::unique_ptr<FATree> current_tree = trySplit(mutation_id, dim, engine, &score);
    if (score > best_score) {
      best_score = score;
      best_tree = std::move(current_tree);
    }
  }
  // Avoiding risk of segfault if something wrong happened before
  if (!best_tree) {
    throw std::logic_error("PolicyMutationLearner::splitMutation: Failed to find a best tree");
  }
  // Updating mutation_candidates
  std::vector<std::unique_ptr<FunctionApproximator>> approximators;
  const Split & split = best_tree->getPreLeafApproximator(space_center).getSplit();
  std::vector<Eigen::MatrixXd> spaces = split.splitSpace(space);
  for (int i = 0; i < split.getNbElements(); i++) {
    MutationCandidate new_mutation;
    new_mutation.space = spaces[i];
    new_mutation.post_training_score = mutation.post_training_score;
    new_mutation.mutation_score = mutation.mutation_score;
    new_mutation.last_training = mutation.last_training;
    new_mutation.is_leaf = true;
    if (i == 0) {
      mutation_candidates[mutation_id] = new_mutation;
    }
    else {
      mutation_candidates.push_back(new_mutation);
    }
  }
  policy_tree = std::move(best_tree);
  policy = buildPolicy(*policy_tree);
}

std::unique_ptr<rosban_fa::FATree>
PolicyMutationLearner::trySplit(int mutation_id, int split_dim,
                                std::default_random_engine * engine,
                                double * score) {
  // Debug message
  std::cout << "\tTesting split along " << split_dim << std::endl;
  // Getting mutation properties
  MutationCandidate mutation = mutation_candidates[mutation_id];
  Eigen::MatrixXd leaf_space = mutation.space;
  Eigen::MatrixXd leaf_center = (leaf_space.col(0) + leaf_space.col(1)) / 2;
  int action_dims = problem->actionDims();
  // Function to train has the following parameters:
  // - Position of the split
  // - Action in each part of the split
  auto parameters_to_approximator =
    [action_dims, split_dim]
    (const Eigen::VectorXd & parameters)
    {
      // Importing values from parameters
      double split_val = parameters[0];
      std::unique_ptr<FunctionApproximator> action0(
        new ConstantApproximator(parameters.segment(1, action_dims)));
      std::unique_ptr<FunctionApproximator> action1(
        new ConstantApproximator(parameters.segment(1+action_dims, action_dims)));
      // Define Function approximator which will replace
      std::unique_ptr<Split> split(new OrthogonalSplit(split_dim, split_val));
      std::vector<std::unique_ptr<FunctionApproximator>> approximators;
      approximators.push_back(std::move(action0));
      approximators.push_back(std::move(action1));
      return std::unique_ptr<FATree>(new FATree(std::move(split), approximators));
    };
  // Defining evaluation function
  rosban_bbo::Optimizer::RewardFunc reward_func =
    [this, split_dim, action_dims, leaf_space, leaf_center, parameters_to_approximator]
    (const Eigen::VectorXd & parameters,
     std::default_random_engine * engine)
    {
      std::unique_ptr<FATree> new_approximator;
      new_approximator = parameters_to_approximator(parameters);
      // Replace approximator
      std::unique_ptr<FATree> new_tree;
      new_tree = policy_tree->copyAndReplaceLeaf(leaf_center, std::move(new_approximator));
      // build policy and evaluate
      // TODO: avoid this second copy of the tree which is not necessary
      std::unique_ptr<Policy> policy = buildPolicy(*new_tree);
      return localEvaluation(*policy, leaf_space, training_evaluations, engine);
    };
  // Define parameters_space
  Eigen::MatrixXd parameters_space(1+2*action_dims,2);
  parameters_space.row(0) = leaf_space.row(split_dim);
  parameters_space.block(1,0,action_dims,2) = problem->getActionLimits();
  parameters_space.block(1+action_dims,0,action_dims,2) = problem->getActionLimits();
  optimizer->setLimits(parameters_space);
  // Computing initial parameters
  // TODO: eventually uses guess from current approximator
  Eigen::VectorXd initial_parameters;
  initial_parameters = (parameters_space.col(0) + parameters_space.col(1)) / 2;
  // Optimize
  Eigen::VectorXd optimized_parameters = optimizer->train(reward_func,
                                                          initial_parameters,
                                                          engine);
  // Make a more approximate evaluation
  std::unique_ptr<FunctionApproximator> optimized_approximator;
  optimized_approximator = parameters_to_approximator(optimized_parameters);
  std::unique_ptr<FATree> splitted_tree;
  splitted_tree = policy_tree->copyAndReplaceLeaf(leaf_center,
                                                  std::move(optimized_approximator));
  std::unique_ptr<Policy> optimized_policy = buildPolicy(*splitted_tree);
  // Evaluate with a larger set
  *score = localEvaluation(*optimized_policy, leaf_space, nb_evaluation_trials, engine);
  // Debug
  std::cout << "\t->Optimized params: " << optimized_parameters.transpose() << std::endl;
  std::cout << "\t->Avg reward: " << *score << std::endl;
  // return optimized approximator
  return std::move(splitted_tree);
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
  rosban_utils::xml_tools::try_read<int>   (node, "training_evaluations", training_evaluations);
  rosban_utils::xml_tools::try_read<double>(node, "split_probability"   , split_probability   );
  // Optimizer is mandatory
  optimizer = rosban_bbo::OptimizerFactory().read(node, "optimizer");
  // Read Policy if provided (optional)
  PolicyFactory().tryRead(node, "policy", policy);
  // Synchronize number of threads
  setNbThreads(nb_threads);
}

std::unique_ptr<Policy> PolicyMutationLearner::buildPolicy(const FATree & tree) {
  std::unique_ptr<FATree> tree_copy(static_cast<FATree *>(tree.clone().release()));
  std::unique_ptr<Policy> result(new FAPolicy(std::move(tree_copy)));
  result->setActionLimits(problem->getActionLimits());
  return std::move(result);
}

}
