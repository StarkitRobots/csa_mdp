#pragma once

#include "rosban_csa_mdp/core/sample.h"
#include "rosban_csa_mdp/solvers/fpf.h"

#include "rosban_regression_forests/core/forest.h"

#include "kd_trees/kd_tree.h"

#include <functional>

/**
 * From Nouri2008: Multi-resolution Exploration in continuous spaces
 */
namespace csa_mdp
{

class MRE {
public:
  /// This class provides some extra functions to KdTree in order to fit the needs of Nouri2008
  class KnownnessTree
  {
  private:
    /// The basic data structure
    kd_trees::KdTree tree;
    /// The maximal number of points by node
    int v;
    /// On which dimension will happen the next split
    int nextSplitDim;
    /// Quick access to the total number of points
    int nbPoints;

  public:
    KnownnessTree(const Eigen::MatrixXd& space, int maxPoints);

    void push(const Eigen::VectorXd& point);

    double getMu() const;

    double getValue(const Eigen::MatrixXd& point) const;
 
  };

  /// MRE needs a modification of the trainingSet creation
  class CustomFPF : public FPF
  {
  public:
    CustomFPF(const Eigen::MatrixXd &q_space,
              int max_points,
              double reward_max);

    /// push a vector (s,a)
    void push(const Eigen::VectorXd &q_point);

  protected:
    virtual regression_forests::TrainingSet
    getTrainingSet(const std::vector<Sample>& samples,
                   std::function<bool(const Eigen::VectorXd&)> is_terminal) override;

  private:
    // Ideally properties should be owned by MRE, but the whole concept needs to be rethought
    double r_max;
    KnownnessTree knownness_tree;
    
  };

private:
  // Configuration
  int plan_period;
  std::function<bool(const Eigen::VectorXd &state)> is_terminal;

  // The transformer from samples to policy
  CustomFPF solver;

  // Acquired samples
  std::vector<Sample> samples;

  // Problem space
  Eigen::MatrixXd state_space;
  Eigen::MatrixXd action_space;

  // Status of current trajectory
  bool active_trajectory;
  Eigen::VectorXd last_state;
  Eigen::VectorXd last_action;
  double last_reward;

  std::default_random_engine random_engine;

  // Quick approach for implementation, yet not generic, force the use of FPF
  std::vector<std::unique_ptr<regression_forests::Forest>> policies;
      

public:
  MRE(const Eigen::MatrixXd& state_space,
      const Eigen::MatrixXd& action_space,
      int max_points,
      double reward_max,
      int plan_period,
      std::function<bool(const Eigen::VectorXd &)> is_terminal);

  /// Feed the learning process with a new tuple (s,a,r), if a trajectory is
  /// in progress, add the sample (old_state, old_action, new_state, new_reward) to the
  /// list of samples, otherwise start a new trajectory with the provided input
  void feed(const Eigen::VectorXd& state,
            const Eigen::VectorXd& action,
            double reward);

  /// Signal to the MRE process to end the current trajectory
  /// i.e. the last seen state and action have no successors
  void endTrajectory();

  /// Return the best action according to current policy
  /// if there is no policy available yet, return a random action
  Eigen::VectorXd getAction(const Eigen::VectorXd &state);

  void updatePolicy();
};

}
