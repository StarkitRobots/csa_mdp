#pragma once

#include "rosban_csa_mdp/solvers/black_box_learner.h"

#include "rosban_bbo/optimizer.h"

#include "rosban_fa/fa_tree.h"

namespace csa_mdp
{

class PolicyMutationLearner : public BlackBoxLearner {
protected:

  /// All the information relative to a mutation candidate are stored in this
  /// structure
  struct MutationCandidate {
    /// Which space is concerned by the candidate
    Eigen::MatrixXd space;
    /// What was the score of this candidate after training
    double post_training_score;
    /// Weight of the mutation in the random selection process
    double mutation_score;
    /// At which iteration was this mutation trained for the last time?
    int last_training;
    /// Is the candidate a leaf or a pre_leaf?
    bool is_leaf;
    /// States met in the current leaf
    std::vector<Eigen::VectorXd> visited_states;
  };

  enum RefinementType {
    wide,
    narrow, 
    local
  };

public:
  PolicyMutationLearner();
  virtual ~PolicyMutationLearner();

  /// Return the number of trials which should be used for evaluation
  int getNbEvaluationTrials() const;

  /// How many evaluations are required inside a single space for evaluation of
  /// the reward during the training phase
  int getTrainingEvaluations() const;

  /// Return the number of calls to the reward function allowed for the optimizer
  int getOptimizerMaxCall() const;

  virtual void init(std::default_random_engine * engine) override;
  virtual void update(std::default_random_engine * engine) override;

  virtual void setNbThreads(int nb_threads) override;

  /// Choose a mutation among the available candidates according to their scores
  int getMutationId(std::default_random_engine * engine);

  /// Evaluate policy and return average reward
  /// If required by internal coniguration, save visited states in
  /// mutation candidates
  /// TODO: complexity should be improved
  double evalAndGetStates(std::default_random_engine * engine);

  /// Update mutation scores according to their properties
  void updateMutationsScores();

  /// Root of the mutation process
  void mutate(int mutation_id, std::default_random_engine * engine);

  /// Mutate a leaf: there is several possibilities:
  /// 1: Refine FunctionApproximator for the leaf
  /// 2: Split the leaf
  void mutateLeaf(int mutation_id, std::default_random_engine * engine);

  /// Mutate a pre-leaf: there is two possibilities:
  /// 1: Change the properties of the split
  /// 2: Contract the split
  void mutatePreLeaf(int mutation_id, std::default_random_engine * engine);

  /// Try to refine the function approximator for the given mutation id
  void refineMutation(int mutation_id,
                      bool change_action,
                      std::default_random_engine * engine);

  RefinementType sampleRefinementType(std::default_random_engine * engine) const;

  /// Split the given mutation on a random dimension
  void splitMutation(int mutation_id, std::default_random_engine * engine);

  /// Try to split along 'split_dim' at the given mutation.
  /// Return the FATree built to replace current approximator and update score
  std::unique_ptr<rosban_fa::FATree>
  trySplit(int mutation_id, int split_dim,
           std::default_random_engine * engine,
           double * score);

  /// Return the defaults parameters for Linear models
  /// Throws an error if fa is not a LinearModel or constant model
  Eigen::VectorXd getGuess(const MutationCandidate & mutation,
                           int action_id) const;

  /// Return the parameters space for training a linear model given the
  /// refinement type
  Eigen::MatrixXd getParametersSpaces(const Eigen::MatrixXd & space,
                                      const Eigen::VectorXd & guess,
                                      RefinementType type,
                                      int action_id) const;

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

  /// Return the best candidate found
  /// rf: the reward function
  /// space: The space allowed for parameters
  /// guess: The initial candidate
  /// engine: Used to draw random numbers
  /// evaluation_gain: Multiplies the usual number of evaluations allowed to the optimizer
  Eigen::VectorXd optimize(rosban_bbo::Optimizer::RewardFunc rf,
                           const Eigen::MatrixXd & space,
                           const Eigen::VectorXd & guess,
                           std::default_random_engine * engine,
                           double evaluation_mult = 1);

  /// Clone the given tree and use it to build a policy. Also set the action
  /// limits
  std::unique_ptr<Policy> buildPolicy(const rosban_fa::FATree & tree);

  /// If 'use_visited_states':
  /// - use states from mutation
  /// Else
  /// - Generate random states
  std::vector<Eigen::VectorXd> getInitialStates(const MutationCandidate & mc,
                                                std::default_random_engine * engine);

  /// Compare the new_tree with current tree. Replace the current tree if the
  /// new tree is better
  /// Return true if the tree has been replaced, false otherwise
  bool submitTree(std::unique_ptr<rosban_fa::FATree> new_tree,
                  const std::vector<Eigen::VectorXd> & initial_states,
                  std::default_random_engine * engine);

protected:
  /// The list of mutations available
  std::vector<MutationCandidate> mutation_candidates;

  /// The current version of the tree
  std::unique_ptr<rosban_fa::FATree> policy_tree;

  /// Current policy
  std::unique_ptr<Policy> policy;

  /// Optimizer used to change split position or to train models
  /// TODO: later, several optimizers should be provided
  std::unique_ptr<rosban_bbo::Optimizer> optimizer;

  /// The number of evaluations used inside each space when training samples
  int training_evaluations;

  /// The increase per iteration of number of evaluations used inside each space
  double training_evaluations_growth;

  /// Probability of splitting a leaf when applying a mutation
  double split_probability;

  /// Probability of testing another action when applying a mutation
  double change_action_probability;

  /// Probability of refining locally when refining
  double local_probability;

  /// Probability of refining with narrow space when refining
  double narrow_probability;

  /// When operating a split on a dimension, this parameters ensures that the
  /// split value is inside [min + delta, max - delta], with:
  /// - delta: (max-min)*split_margin
  /// split_margin has to be in [0, 0.5[ (otherwise, split space is empty)
  double split_margin;

  /// When > 0, the number of trajectories evaluated at each step by optimizer
  /// is equal to 'evaluations_ratio' times the number of trajectories used to
  /// compute the average reward over the whole space
  /// TODO: implement it in pml
  double evaluations_ratio;

  /// Growth of the number of evaluation trials at each step
  double evaluations_growth;

  /// age_score = age_basis ^ (time_since_update)
  double age_basis;

  /// If enabled, then maximal coefficients are always determined according to
  /// the whole problem space
  bool avoid_growing_slopes;

  /// If true, then only visited states can be used for optimization
  /// If false, states are sampled randomly inside the hyper-rectangle
  bool use_visited_states;

  /// Is density of visited states used to score mutation candidates?
  bool use_density_score;
};

}
