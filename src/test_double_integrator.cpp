#include "rosban_csa_mdp/core/problem.h"

#include "rosban_csa_mdp/solvers/extra_trees.h"

using csa_mdp::ExtraTrees;
using csa_mdp::Problem;

class DoubleIntegrator : public Problem
{
public:
  DoubleIntegrator()
    {
      Eigen::MatrixXd state_limits(2,2), action_limits(1,2);
      state_limits << -1, 1, -1, 1;
      action_limits << -1, 1;
      setStateLimits(state_limits);
      setActionLimits(action_limits);
    }

  bool isTerminal(const Eigen::VectorXd & state) const override
    {
      for (int i = 0; i < 2; i++)
      {
        if (state(i) < state(i,0) || state(i) > state(i,1))
          return false;
      }
      return true;
    }

  double getReward(const Eigen::VectorXd & state,
                   const Eigen::VectorXd & action,
                   const Eigen::VectorXd & dst) override
    {
      if (isTerminal(dst)) {
        return -50;
      }
      double posCost = dst(0) * dst(0);
      double accCost = action(0) * action(0);
      return -(posCost + accCost);
    }

  Eigen::VectorXd getSuccessor(const Eigen::VectorXd & state,
                               const Eigen::VectorXd & action) override
    {
      double integrationStep = 0.05;
      double simulationStep = 0.5;
      double elapsed = 0;
      Eigen::VectorXd currentState = state;
      while (elapsed < simulationStep)
      {
        double dt = std::min(simulationStep - elapsed, integrationStep);
                                               
        Eigen::Vector2d nextState;

        double vel = currentState(1);
        double acc = action(0);
        nextState(0) = currentState(0) + dt * vel;
        nextState(1) = currentState(1) + dt * acc;
        elapsed += dt;
        currentState = nextState;
      }
      return currentState;
    }
};

int main()
{
  DoubleIntegrator di;
  Eigen::VectorXd initial_state(2);
  initial_state << 1, 0;
  int trajectory_max_length = 200;
  int nb_trajectories = 500;
  std::vector<csa_mdp::Sample> mdp_samples = di.getRandomBatch(initial_state,
                                                               trajectory_max_length,
                                                               nb_trajectories);
  ExtraTrees::Config conf;
  conf.horizon = 10;
  conf.discount = 0.98;
  conf.preFilter = false;
  conf.parallelMerge =true;
  conf.maxActionTiles = 100;
  conf.ETConf.k = 3;
  conf.ETConf.nMin = 1;
  conf.ETConf.nbTrees = 25;
  conf.ETConf.minVar = std::pow(10, -6);
  conf.ETConf.apprType = regression_forests::ApproximationType::PWC;
  ExtraTrees solver(di.getStateLimits(),
                    di.getActionLimits());
  auto is_terminal = [di](const Eigen::VectorXd &state){return di.isTerminal(state);};
  solver.solve(mdp_samples, conf, is_terminal);
  solver.valueForest().save("/tmp/test_di.data");

//  EvaluationConfig config;
//  config.forestPolicy = true;
//  config.maxLeafs = 250;
//  config.pConf.nbSamples     = 10000;
//  config.pConf.preFilter     = false;
//  config.pConf.parallelMerge = true;
//  config.pConf.ETConf.k         = 2;
//  config.pConf.ETConf.nMin      = 1500;//PWC: ~130 | PWL: ~1500
//  config.pConf.ETConf.nbTrees   = 25;
//  config.pConf.ETConf.minVar    = std::pow(10, -8);
//  config.pConf.ETConf.bootstrap = false;
//  config.pConf.ETConf.apprType  = ApproximationType::PWL;
//  Math::Problems::DoubleIntegrator di;
}
