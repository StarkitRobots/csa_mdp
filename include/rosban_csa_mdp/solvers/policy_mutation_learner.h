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
  };

public:
  PolicyMutationLearner();
  virtual ~PolicyMutationLearner();

  virtual void init(std::default_random_engine * engine) override;
  virtual void update(std::default_random_engine * engine) override;

  virtual void setNbThreads(int nb_threads) override;

  /// Choose a mutation among the available candidates according to their scores
  int getMutationId(std::default_random_engine * engine);

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
  void refineMutation(int mutation_id, std::default_random_engine * engine);

  /// Split the given mutation on a random dimension
  void splitMutation(int mutation_id, std::default_random_engine * engine);

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

  /// Clone the given tree and use it to build a policy. Also set the action
  /// limits
  std::unique_ptr<Policy> buildPolicy(const rosban_fa::FATree & tree);

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

  /// Probability of splitting a leaf when applying a mutation
  double split_probability;  
};

}
