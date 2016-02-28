#include "rosban_csa_mdp/core/problem.h"

#include "rosban_csa_mdp/solvers/fpf.h"

using csa_mdp::FPF;
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
        if (state(i) < getStateLimits()(i,0) || state(i) > getStateLimits()(i,1))
          return true;
      }
      return false;
    }

  double getReward(const Eigen::VectorXd & state,
                   const Eigen::VectorXd & action,
                   const Eigen::VectorXd & dst) override
    {
      (void) state;
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
  int nb_trajectories = 200;
  std::vector<csa_mdp::Sample> mdp_samples = di.getRandomBatch(initial_state,
                                                               trajectory_max_length,
                                                               nb_trajectories);
  FPF::Config conf;
  conf.setStateLimits(di.getStateLimits());
  conf.setActionLimits(di.getActionLimits());
  conf.horizon = 10;
  conf.discount = 0.98;
  conf.max_action_tiles = 40;
  conf.q_value_conf.k = 3;
  conf.q_value_conf.n_min = 1;
  conf.q_value_conf.nb_trees = 25;
  conf.q_value_conf.min_var = std::pow(10, -4);
  conf.q_value_conf.appr_type = regression_forests::ApproximationType::PWC;
  conf.policy_samples = 10000;
  conf.policy_conf.k = 2;
  conf.policy_conf.n_min = 1500;
  conf.policy_conf.nb_trees = 25;
  conf.policy_conf.min_var = std::pow(10, -4);
  conf.policy_conf.appr_type = regression_forests::ApproximationType::PWL;
  FPF solver;
  auto is_terminal = [di](const Eigen::VectorXd &state){return di.isTerminal(state);};
  solver.solve(mdp_samples, is_terminal, conf);
  solver.getValueForest().save("/tmp/test_di_values.data");
  solver.getPolicyForest(0).save("/tmp/test_di_policy.data");
}
