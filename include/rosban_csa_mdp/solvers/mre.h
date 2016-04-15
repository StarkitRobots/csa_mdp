#pragma once

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

class MRE {
public:
  class Config : public rosban_utils::Serializable
  {
  public:
    Config();

    std::string class_name() const override;
    void to_xml(std::ostream &out) const override;
    void from_xml(TiXmlNode *node) override;

    int plan_period;
    MREFPF::Config mrefpf_conf;
    KnownnessForest::Config knownness_conf;
  };

public:
  MRE(const Config &conf,
      std::function<bool(const Eigen::VectorXd &)> is_terminal);

  /// Feed the learning process with a new sample, update policy if required
  void feed(const Sample &s);

  /// Return the best action according to current policy
  /// if there is no policy available yet, return a random action
  Eigen::VectorXd getAction(const Eigen::VectorXd &state);

  /// Called automatically on feed each plan_period samples
  void updatePolicy();

  const regression_forests::Forest & getPolicy(int dim);

  void savePolicies(const std::string &prefix);
  void saveValue(const std::string &prefix);
  void saveKnownnessTree(const std::string &prefix);
  void saveStatus(const std::string &prefix);

  double getQValueTrainingSetTime() const;
  double getQValueExtraTreesTime()  const;
  double getPolicyTrainingSetTime() const;
  double getPolicyExtraTreesTime()  const;

  const Eigen::MatrixXd & getStateSpace();
  const Eigen::MatrixXd & getActionSpace();

private:
  /// MRE Configuration
  Config conf;

  /// Solver
  MREFPF solver;

  /// Knownness Forest
  std::shared_ptr<KnownnessForest> knownness_forest;

  /// Terminal function
  std::function<bool(const Eigen::VectorXd &state)> is_terminal;

  /// Acquired samples until now
  std::vector<Sample> samples;

  /// Random generator
  std::default_random_engine random_engine;

  // Quick approach for implementation, yet not generic, force the use of FPF
  std::vector<std::unique_ptr<regression_forests::Forest>> policies;
};

}
