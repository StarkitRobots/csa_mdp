#include "rosban_csa_mdp/core/problem.h"

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

  bool isTerminal(const Eigen::VectorXd & state) override
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
}
