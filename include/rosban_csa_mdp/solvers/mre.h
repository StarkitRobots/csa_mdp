#pragma once

#include "rosban_csa_mdp/core/sample.h"
#include "rosban_csa_mdp/solvers/fpf.h"

#include "rosban_regression_forests/core/forest.h"

#include "kd_trees/kd_tree.h"

#include <functional>
#include <random>

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
    /// Random generator for the Random Type
    std::default_random_engine random_engine;

  public:

    enum Type
    {
      // Follows MRE description
      Original,
      // Splitting on median of the biggest dimension (using rescaling)
      Test,
      // Split on a random dimension at a random position (between min and max found)
      // Value is based on dimension product
      Random
    };
    /// Which version of knownness tree is activated?
    Type type;

    KnownnessTree(const Eigen::MatrixXd& space, int maxPoints, Type type);

    void push(const Eigen::VectorXd& point);

    double getMu() const;

    double getValue(const Eigen::VectorXd& point) const;
    double getValue(const Eigen::MatrixXd& space, int nb_points) const;

    regression_forests::Node * convertToRegNode(const kd_trees::KdNode *node,
                                                Eigen::MatrixXd &space) const;
    std::unique_ptr<regression_forests::Tree> convertToRegTree() const;
 
  };

  /// MRE needs a modification of the trainingSet creation
  class CustomFPF : public FPF
  {
  public:
    CustomFPF(const Eigen::MatrixXd &q_space,
              int max_points,
              double reward_max,
              int nb_trees,
              KnownnessTree::Type type);

    /// push a vector (s,a)
    void push(const Eigen::VectorXd &q_point);

    double getKnownness(const Eigen::VectorXd& point) const;

    std::unique_ptr<regression_forests::Forest> getKnownnessForest();

  protected:
    virtual regression_forests::TrainingSet
    getTrainingSet(const std::vector<Sample>& samples,
                   std::function<bool(const Eigen::VectorXd&)> is_terminal) override;

    // Custom way of chosing reward states
    virtual std::vector<Eigen::VectorXd>
    getPolicyTrainingStates(const std::vector<Sample>& samples) override;


  public:
    // Ideally properties should be owned by MRE, but the whole concept needs to be rethought
    std::vector<KnownnessTree> knownness_forest;
    double r_max;
    
  };

private:
  // Configuration
  /// If plan_period is less than 0, then caller has to call updatePolicy explicitely
  int plan_period;
  std::function<bool(const Eigen::VectorXd &state)> is_terminal;

  // The transformer from samples to policy
  CustomFPF solver;

  // Acquired samples
  std::vector<Sample> samples;

  // Problem space
  Eigen::MatrixXd state_space;
  Eigen::MatrixXd action_space;

  std::default_random_engine random_engine;

  // Quick approach for implementation, yet not generic, force the use of FPF
  std::vector<std::unique_ptr<regression_forests::Forest>> policies;
      

public:
  MRE(const Eigen::MatrixXd& state_space,
      const Eigen::MatrixXd& action_space,
      int max_points,
      double reward_max,
      int plan_period,
      int nb_trees,
      KnownnessTree::Type knownness_tree_type,
      const FPF::Config &fpf_conf,
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

  double getQValueTime() const;
  double getPolicyTime() const;
};

}
