#pragma once

#include "rosban_csa_mdp/solvers/learner.h"

#include "rosban_csa_mdp/core/sample.h"
#include "rosban_csa_mdp/solvers/mre_fpf.h"
#include "rosban_csa_mdp/knownness/knownness_forest.h"

#include "rosban_regression_forests/core/forest.h"

#include "kd_trees/kd_tree.h"

#include <functional>
#include <random>

/**
 * From Nouri2009: Multi-resolution Exploration in continuous spaces
 */
namespace csa_mdp
{

class MRE : public Learner{
public:
  MRE();

  /// Also update internal structure
  virtual void setNbThreads(int nb_threads) override;

  /// Feed the learning process with a new sample, update policy if required
  void feed(const Sample &s) override;

  /// Return the best action according to current policy
  /// if there is no policy available yet, return a random action
  Eigen::VectorXd getAction(const Eigen::VectorXd &state) override;

  /// Update policy and solver status
  /// Called automatically on feed each plan_period samples
  void internalUpdate() override;

  /// While policy has not been updated with enough samples, this is false and actions
  /// are chosen at uniformous random
  bool hasAvailablePolicy();

  const regression_forests::Forest & getPolicy(int dim);

  void savePolicy(const std::string &prefix) override;
  void saveValue(const std::string &prefix);
  void saveKnownnessTree(const std::string &prefix);
  void saveStatus(const std::string &prefix) override;

  void setStateLimits(const Eigen::MatrixXd & limits) override;
  void setActionLimits(const Eigen::MatrixXd & limits) override;
  void updateQSpaceLimits();

  std::string class_name() const override;
  void to_xml(std::ostream &out) const override;
  void from_xml(TiXmlNode *node) override;

private:
  /// Which is the plan frequency: '-1' -> update only when requested
  int plan_period;
  /// Configuration used for the solver
  MREFPF::Config mrefpf_conf;
  /// Configuration used for the knownness function
  KnownnessForest::Config knownness_conf;

  /// Solver
  MREFPF solver;

  /// Knownness Forest
  std::shared_ptr<KnownnessForest> knownness_forest;

  /// Acquired samples until now
  std::vector<Sample> samples;

  /// Random generator
  std::default_random_engine random_engine;

  // Quick approach for implementation, yet not generic, force the use of FPF
  std::vector<std::unique_ptr<regression_forests::Forest>> policies;
};

}
