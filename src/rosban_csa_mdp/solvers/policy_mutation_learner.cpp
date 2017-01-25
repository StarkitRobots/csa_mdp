#include "rosban_csa_mdp/solvers/policy_mutation_learner.h"

#include "rosban_bbo/optimizer_factory.h"

#include "rosban_csa_mdp/core/fa_policy.h"
#include "rosban_csa_mdp/core/policy_factory.h"

#include "rosban_fa/constant_approximator.h"
#include "rosban_fa/fake_split.h"
#include "rosban_fa/function_approximator_factory.h"
#include "rosban_fa/linear_approximator.h"
#include "rosban_fa/orthogonal_split.h"
#include "rosban_random/tools.h"
#include "rosban_utils/time_stamp.h"

using namespace rosban_fa;
using rosban_utils::TimeStamp;

namespace csa_mdp
{

PolicyMutationLearner::PolicyMutationLearner()
  : training_evaluations(50),
    training_evaluations_growth(0),
    split_probability(0.1),
    local_probability(0.2),
    narrow_probability(0.2),
    split_margin(0.2),
    evaluations_ratio(-1),
    evaluations_growth(0),
    avoid_growing_slopes(true),
    shared_initial_states(true)
{
}

PolicyMutationLearner::~PolicyMutationLearner() {}

int PolicyMutationLearner::getNbEvaluationTrials() const {
  return nb_evaluation_trials + (int)(evaluations_growth * iterations);
}

int PolicyMutationLearner::getTrainingEvaluations() const {
  return training_evaluations + (int)(training_evaluations_growth * iterations);
}

int PolicyMutationLearner::getOptimizerMaxCall() const {
  int evaluations_allowed = evaluations_ratio * getNbEvaluationTrials();
  return (int)(evaluations_allowed / getTrainingEvaluations());
}


void PolicyMutationLearner::init(std::default_random_engine * engine) {
  // If a policy has been specified, try to extract a FATree from policy
  if (policy) {
    policy_tree = policy->extractFATree();
    policy = buildPolicy(*policy_tree);
    std::vector<Eigen::MatrixXd> leaf_spaces;
    policy_tree->addSpaces(problem->getStateLimits(), &leaf_spaces);
    for (const Eigen::MatrixXd & leaf_space : leaf_spaces) {
      MutationCandidate candidate;
      candidate.space = leaf_space;
      candidate.mutation_score = 1.0;
      candidate.last_training = 0;
      candidate.is_leaf = true;
      mutation_candidates.push_back(candidate);
    }
  }
  else {
    std::unique_ptr<Split> split(new FakeSplit());
    // Default action choice is 0
    int action_id = 0;
    const Eigen::MatrixXd & action_limits = problem->getActionLimits(action_id);
    // Default action is the middle of the space
    Eigen::VectorXd action(action_limits.rows() + 1);
    action(0) = action_id;
    action.segment(1, action_limits.rows()) = (action_limits.col(0) + action_limits.col(1)) / 2;
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
                                                    getNbEvaluationTrials(),
                                                    engine);
    candidate.mutation_score = 1.0;
    candidate.last_training = 0;
    candidate.is_leaf = true;
    mutation_candidates.push_back(candidate);
  }
  double avg_reward = evaluatePolicy(*policy, getNbEvaluationTrials(), engine);
  std::cout << "Initial Reward: " << avg_reward << std::endl;
}

void PolicyMutationLearner::update(std::default_random_engine * engine) {
  TimeStamp start = TimeStamp::now();
  int mutation_id = getMutationId(engine);
  mutate(mutation_id, engine);
  TimeStamp post_mutation = TimeStamp::now();
  double new_reward = evaluatePolicy(*policy, getNbEvaluationTrials(), engine);
  TimeStamp post_evaluation = TimeStamp::now();
  std::cout << "New reward: " << new_reward << std::endl;
  policy_tree->save("policy_tree.bin");
  updateMutationsScores();
  TimeStamp post_misc = TimeStamp::now();
  // Writing data
  writeTime("mutation"  , diffSec(start          , post_mutation  ));
  writeTime("evaluation", diffSec(post_mutation  , post_evaluation));
  writeTime("misc"      , diffSec(post_evaluation, post_misc      ));
  writeScore(new_reward);
}

Eigen::VectorXd PolicyMutationLearner::optimize(rosban_bbo::Optimizer::RewardFunc rf,
                                                const Eigen::MatrixXd & space,
                                                const Eigen::VectorXd & guess,
                                                std::default_random_engine * engine,
                                                double evaluation_mult) {
  // If activated, set the maximal number of calls to the reward function to optimizer
  if (evaluations_ratio > 0) {
    optimizer->setMaxCalls(getOptimizerMaxCall() * evaluation_mult);
  }
  optimizer->setLimits(space);
  return optimizer->train(rf, guess, engine);
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

void PolicyMutationLearner::updateMutationsScores() {
  for (MutationCandidate & c : mutation_candidates) {
    // TODO: add custom parameter for basis
    c.mutation_score = pow(1.02, (iterations - c.last_training));
  }
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

void PolicyMutationLearner::mutateLeaf(int mutation_id,
                                       std::default_random_engine * engine) {
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
  RefinementType type = sampleRefinementType(engine);
  // Get reference to the appropriate mutation
  MutationCandidate * mutation = &(mutation_candidates[mutation_id]);
  // Get space, and space center
  Eigen::MatrixXd space = mutation->space;
  int input_dim = problem->stateDims();
  Eigen::VectorXd space_center = (space.col(0) + space.col(1)) / 2;
  // Getting current action_id
  Eigen::VectorXd current_action = policy_tree->predict(space_center);
  int action_id = (int)current_action(0);
  int output_dim = problem->actionDims(action_id);
  Eigen::MatrixXd action_limits = problem->getActionLimits(action_id);
  // Debug:
  std::cout << "-> Applying a refine mutation of type ";
  std::string name;
  switch(type) {
    case RefinementType::local:
      std::cout << "local";
      break;
    case RefinementType::narrow:
      std::cout << "narrow";
      break;
    case RefinementType::wide:
      std::cout << "wide";
      break;
  }
  std::cout << " on space" << std::endl
            << space.transpose() << std::endl;
  // Are we using a single set of initial_states per mutation?
  std::vector<Eigen::VectorXd> initial_states;
  int nb_evaluations_allowed = getTrainingEvaluations();
  if (shared_initial_states) {
    initial_states = rosban_random::getUniformSamples(space, nb_evaluations_allowed, engine);
  }
  // Add the parameters to provide an action_id
  auto parameters_to_full =
    [input_dim, output_dim, action_id]
    (const Eigen::VectorXd & raw_params)
    {
      Eigen::VectorXd full_params((input_dim+1) * (output_dim+1));
      for (int col = 0; col < input_dim+1; col++) {
        int start = col * (output_dim+1);
        full_params(start) = col == 0 ? action_id : 0;
        full_params.segment(start+1, output_dim) = raw_params.segment(col*output_dim, output_dim);
      }
      return full_params;
    };
  // Training function
  // TODO: use other models than PWL
  rosban_bbo::Optimizer::RewardFunc reward_func =
    [this, &space, input_dim, output_dim, &space_center, &initial_states,
     nb_evaluations_allowed, action_id, &parameters_to_full]
    (const Eigen::VectorXd & parameters,
     std::default_random_engine * engine)
    {
      // Add the parameters to choose action id
      Eigen::VectorXd full_params = parameters_to_full(parameters);
      // Build the tree
      std::unique_ptr<FunctionApproximator> new_approximator(
        new LinearApproximator(input_dim, output_dim + 1, full_params, space_center));
      std::unique_ptr<FATree> new_tree;
      new_tree = policy_tree->copyAndReplaceLeaf(space_center, std::move(new_approximator));
      //TODO: avoid this second copy of the tree which is not necessary
      std::unique_ptr<Policy> policy = buildPolicy(*new_tree);
      // Type of evaluation depends on the fact that initial_states have been created
      if (initial_states.size() > 0) {
        return evaluation(*policy, initial_states, engine);
      }
      return localEvaluation(*policy, space, nb_evaluations_allowed, engine);
    };
  // Debug function
  auto parameters_to_matrix =
    [input_dim, output_dim](const Eigen::VectorXd & parameters)
    {
      Eigen::MatrixXd result(output_dim + 1, input_dim+1);
      for (int col = 0; col < input_dim+1; col++) {
        result.col(col) = parameters.segment(col * output_dim, output_dim);
      }
      return result;
    };
  // Getting initial guess
  Eigen::VectorXd parameters_guess = getGuess(*mutation);
  // Getting parameters_space
  Eigen::MatrixXd parameters_space = getParametersSpaces(space,
                                                         parameters_guess,
                                                         type,
                                                         action_id);
  for (int dim = 0; dim < parameters_guess.rows(); dim++) {
    double original = parameters_guess(dim);
    double min = parameters_space(dim, 0);
    double max = parameters_space(dim, 1);
    parameters_guess(dim) = std::min(max,std::max(min,original));
  }
  std::cout << "Parameters_space:" << std::endl
            << parameters_space << std::endl;
  std::cout << "Guess:" << std::endl
            << parameters_to_matrix(parameters_guess) << std::endl;

  // Creating refined approximator
  Eigen::VectorXd refined_parameters = optimize(reward_func,
                                                parameters_space,
                                                parameters_guess,
                                                engine);
  refined_parameters = parameters_to_full(refined_parameters);
  std::cout << "\tRefined parameters:" << std::endl
            << parameters_to_matrix(refined_parameters) << std::endl;

  std::unique_ptr<FunctionApproximator> refined_approximator(
    new LinearApproximator(input_dim, output_dim + 1, refined_parameters, space_center));
  // Creating refined policy
  std::unique_ptr<FATree> refined_tree;
  refined_tree = policy_tree->copyAndReplaceLeaf(space_center, std::move(refined_approximator));
  std::unique_ptr<Policy> refined_policy = buildPolicy(*refined_tree);
  // Evaluate initial and refined policy with updated parameters (on local space)
  double initial_reward = localEvaluation(*policy, space, getNbEvaluationTrials(), engine);
  double refined_reward = localEvaluation(*refined_policy, space, getNbEvaluationTrials(), engine);
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

PolicyMutationLearner::RefinementType
PolicyMutationLearner::sampleRefinementType(std::default_random_engine * engine) const {
  double val = std::uniform_real_distribution<double>(0,1)(*engine);
  val -= local_probability;
  if (val < 0) return RefinementType::local;
  val -= narrow_probability;
  if (val < 0) return RefinementType::narrow;
  return RefinementType::wide;
}

void PolicyMutationLearner::splitMutation(int mutation_id,
                                          std::default_random_engine * engine) {
  const MutationCandidate & mutation = mutation_candidates[mutation_id];
  Eigen::MatrixXd space = mutation.space;
  Eigen::VectorXd space_center = (space.col(0) + space.col(1)) / 2;
  const FunctionApproximator & leaf_fa = policy_tree->getLeafApproximator(space_center);
  // Debug message
  std::cout << "-> Applying a split mutation on space" << std::endl
            << space.transpose() << std::endl;
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
  // Getting ref to the chosen Split
  const Split & split = best_tree->getPreLeafApproximator(space_center).getSplit();
  // Updating mutation_candidates
  std::vector<std::unique_ptr<FunctionApproximator>> approximators;
  std::vector<Eigen::MatrixXd> spaces = split.splitSpace(space);
  for (int i = 0; i < split.getNbElements(); i++) {
    MutationCandidate new_mutation;
    new_mutation.space = spaces[i];
    new_mutation.post_training_score = mutation.post_training_score;
    new_mutation.mutation_score = mutation.mutation_score;
    new_mutation.last_training = 0;
    new_mutation.is_leaf = true;
    if (i == 0) {
      mutation_candidates[mutation_id] = new_mutation;
    }
    else {
      mutation_candidates.push_back(new_mutation);
    }
  }
  // Evaluating policy with respect to previous solution
  std::unique_ptr<Policy> proposed_policy = buildPolicy(*best_tree);
  double current_reward = localEvaluation(*policy, space,
                                          getNbEvaluationTrials(), engine);
  double proposed_reward = localEvaluation(*proposed_policy, space,
                                          getNbEvaluationTrials(), engine);
  std::cout << "\tCurrent reward: " << current_reward << std::endl;
  std::cout << "\tProposed reward: " << proposed_reward << std::endl;
  // If split improved reward, keep the new tree, otherwise, use previous
  // approximators
  if (proposed_reward > current_reward) {
    policy_tree = std::move(best_tree);
    policy = buildPolicy(*policy_tree);
  }
  else {
    // Create a new FATree with cloned approximators
    std::unique_ptr<Split> split_copy = split.clone();
    std::vector<std::unique_ptr<FunctionApproximator>> approximators;
    for (int elem = 0; elem < split.getNbElements(); elem++) {
      approximators.push_back(leaf_fa.clone());
    }
    std::unique_ptr<FATree> new_leaf_fa(new FATree(std::move(split_copy),
                                                   approximators));
    // Replace it in policy tree and update policy
    policy_tree->replaceApproximator(space_center, std::move(new_leaf_fa));
    policy = buildPolicy(*policy_tree);
  }
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
  // Getting current action_id
  Eigen::VectorXd current_action = policy_tree->predict(leaf_center);
  int action_id = (int)current_action(0);
  Eigen::MatrixXd action_limits = problem->getActionLimits(action_id);
  int action_dims = problem->actionDims(action_id);
  // Function to train has the following parameters:
  // - Position of the split
  // - Action in each part of the split
  auto parameters_to_approximator =
    [action_dims, split_dim, action_id]
    (const Eigen::VectorXd & parameters)
    {
      // Importing values from parameters
      double split_val = parameters[0];
      // Adding action_id
      Eigen::VectorXd params0 = Eigen::VectorXd::Zero(action_dims+1);
      params0(0) = action_id;
      params0.segment(1,action_dims) = parameters.segment(1, action_dims);
      Eigen::VectorXd params1 = Eigen::VectorXd::Zero(action_dims+1);
      params1(0) = action_id;
      params1.segment(1,action_dims) = parameters.segment(1 + action_dims, action_dims);
      // Building FunctionApproximator
      std::unique_ptr<FunctionApproximator> action0(new ConstantApproximator(params0));
      std::unique_ptr<FunctionApproximator> action1(new ConstantApproximator(params1));
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
      return localEvaluation(*policy, leaf_space, getTrainingEvaluations(), engine);
    };
  // Computing boundaries for split
  double dim_min = leaf_space(split_dim,0);
  double dim_max = leaf_space(split_dim,1);
  double delta = (dim_max - dim_min) * split_margin;
  double split_min = dim_min + delta;
  double split_max = dim_max - delta;
  std::cout << "\t->Split value range:  [" << split_min << ", " << split_max << "]"
            << std::endl;
  // Define parameters_space
  Eigen::MatrixXd parameters_space(1+2*action_dims,2);
  parameters_space(0,0) = split_min;
  parameters_space(0,1) = split_max;
  parameters_space.block(1,0,action_dims,2) = action_limits;
  parameters_space.block(1+action_dims,0,action_dims,2) = action_limits;
  // Computing initial parameters using current approximator
  double split_default_val = (split_min + split_max) / 2;
  std::unique_ptr<Split> initial_split(new OrthogonalSplit(split_dim, split_default_val));
  std::vector<Eigen::MatrixXd> split_spaces = initial_split->splitSpace(leaf_space);
  Eigen::VectorXd initial_parameters(1+2*action_dims);
  initial_parameters(0) = split_default_val;
  for (size_t leaf_id = 0; leaf_id < split_spaces.size(); leaf_id++) {
    int start = 1 + action_dims * leaf_id;
    const Eigen::MatrixXd & curr_space = split_spaces[leaf_id];
    Eigen::VectorXd center = (curr_space.col(0) + curr_space.col(1)) / 2;
    initial_parameters.segment(start, action_dims) = policy_tree->predict(center);
  }
  // Optimize
  double evaluation_ratio = 1.0 / problem->getStateLimits().rows();
  Eigen::VectorXd optimized_parameters = optimize(reward_func,
                                                  parameters_space,
                                                  initial_parameters,
                                                  engine,
                                                  evaluation_ratio);
  // Make a more approximate evaluation
  std::unique_ptr<FunctionApproximator> optimized_approximator;
  optimized_approximator = parameters_to_approximator(optimized_parameters);
  std::unique_ptr<FATree> splitted_tree;
  splitted_tree = policy_tree->copyAndReplaceLeaf(leaf_center,
                                                  std::move(optimized_approximator));
  std::unique_ptr<Policy> optimized_policy = buildPolicy(*splitted_tree);
  // Evaluate with a larger set
  *score = localEvaluation(*optimized_policy, leaf_space,
                           evaluation_ratio * getNbEvaluationTrials(), engine);
  // Debug
  std::cout << "\t->Optimized params: " << optimized_parameters.transpose() << std::endl;
  std::cout << "\t->Avg reward: " << *score << std::endl;
  // return optimized approximator
  return std::move(splitted_tree);
}

Eigen::VectorXd PolicyMutationLearner::getGuess(const MutationCandidate & mutation) const {
  // Get space, center and original FA
  Eigen::MatrixXd space = mutation.space;
  Eigen::VectorXd space_center = (space.col(0) + space.col(1)) / 2;
  const FunctionApproximator & fa = policy_tree->getLeafApproximator(space_center);
  int action_id = fa.predict(space_center)(0);
  Eigen::MatrixXd action_limits = problem->getActionLimits(action_id);
  int action_dims = problem->actionDims(action_id);
  // Default parameters
  Eigen::VectorXd guess = LinearApproximator::getDefaultParameters(space,
                                                                   action_limits);
  // Case of Constant Approximator
  try {
    const ConstantApproximator & approximation =
      dynamic_cast<const ConstantApproximator &>(fa);
    guess.segment(0, action_dims) = approximation.getValue();
    return guess;
  } catch (const std::bad_cast & exc) {
    // Nothing to be done
  }
  // Case of Linear Approximator
  try {
    const LinearApproximator & approximation =
      dynamic_cast<const LinearApproximator &>(fa);
    Eigen::VectorXd bias_at_center = approximation.getBias(space_center);
    Eigen::VectorXd coeffs = approximation.getCoeffs();
    guess.segment(0, action_dims) = bias_at_center;
    guess.segment(action_dims, coeffs.rows()) = coeffs;
    return guess;
  } catch (const std::bad_cast & exc) {
    // Nothing to be done
  }
  throw std::logic_error("PolicyMutationLearner::getGuess: type of 'fa' is not handled");
}

Eigen::MatrixXd PolicyMutationLearner::getParametersSpaces(const Eigen::MatrixXd & space,
                                                           const Eigen::VectorXd & guess,
                                                           RefinementType type,
                                                           int action_id) const {
  // Which space is used for parameters
  Eigen::MatrixXd parameters_space;
  bool narrow_slope = (type != RefinementType::wide);
  Eigen::MatrixXd space_for_parameters = space;
  if (avoid_growing_slopes) {
    space_for_parameters = problem->getStateLimits();
  }
  parameters_space = LinearApproximator::getParametersSpace(space_for_parameters,
                                                            problem->getActionLimits(action_id),
                                                            narrow_slope);
  // If type is local, then: search around guess
  if (type == RefinementType::local) {
    Eigen::VectorXd parameters_size = parameters_space.col(1) - parameters_space.col(0);
    parameters_space.col(0) = guess - parameters_size / 2;
    parameters_space.col(1) = guess + parameters_size / 2;
  }
  return parameters_space;
}

std::string PolicyMutationLearner::class_name() const {
  return "PolicyMutationLearner";
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
  rosban_utils::xml_tools::try_read<bool>  (node, "avoid_growing_slopes" , avoid_growing_slopes );
  rosban_utils::xml_tools::try_read<bool>  (node, "shared_initial_states", shared_initial_states);
  rosban_utils::xml_tools::try_read<int>   (node, "training_evaluations" , training_evaluations );
  rosban_utils::xml_tools::try_read<double>(node, "training_evaluations_growth",
                                            training_evaluations_growth);
  rosban_utils::xml_tools::try_read<double>(node, "split_probability"   , split_probability   );
  rosban_utils::xml_tools::try_read<double>(node, "local_probability"   , local_probability   );
  rosban_utils::xml_tools::try_read<double>(node, "narrow_probability"  , narrow_probability  );
  rosban_utils::xml_tools::try_read<double>(node, "split_margin"        , split_margin        );
  rosban_utils::xml_tools::try_read<double>(node, "evaluations_ratio"   , evaluations_ratio   );
  rosban_utils::xml_tools::try_read<double>(node, "evaluations_growth"  , evaluations_growth  );
  // Optimizer is mandatory
  optimizer = rosban_bbo::OptimizerFactory().read(node, "optimizer");
  // Read Policy if provided (optional)
  PolicyFactory().tryRead(node, "policy", policy);
  // Performing some checks
  if (split_margin < 0 || split_margin >= 0.5) {
    throw std::logic_error("PolicyMutationLearner::from_xml: invalid value for split_margin");
  }
  //TODO: add checks on probability
  // Synchronize number of threads
  setNbThreads(nb_threads);
}

std::unique_ptr<Policy> PolicyMutationLearner::buildPolicy(const FATree & tree) {
  std::unique_ptr<FATree> tree_copy(static_cast<FATree *>(tree.clone().release()));
  std::unique_ptr<Policy> result(new FAPolicy(std::move(tree_copy)));
  result->setActionLimits(problem->getActionsLimits());
  return std::move(result);
}

}
