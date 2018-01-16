#include "rosban_csa_mdp/online_planners/open_loop_planner.h"

#include "rosban_bbo/optimizer_factory.h"

namespace csa_mdp
{

OpenLoopPlanner::OpenLoopPlanner()
  : look_ahead(0), rollouts_per_sample(1), discount(1)
{
}

Eigen::VectorXd
OpenLoopPlanner::planNextAction(const Problem & p,
                                const Eigen::VectorXd & state,
                                const rosban_fa::FunctionApproximator & value_function,
                                std::default_random_engine * engine)
{
  // Consistency checks
  if (p.getNbActions() != 1) {
    throw std::runtime_error("OpenLoopPlanner::planNextAction: no support for hybrid action spaces");
  }
  if (look_ahead <= 0) {
    throw std::runtime_error("OpenLoopPlanner::planNextAction: invalid value for look_ahead: '"
                             + std::to_string(look_ahead) + "'");
  }
  // Computing optimizer space
  Eigen::MatrixXd action_limits = p.getActionLimits(0);
  int action_dims = action_limits.rows();
  Eigen::MatrixXd optimizer_limits(action_dims*look_ahead,2);
  for (int action = 0; action < look_ahead; action++) {
    optimizer_limits.block(action* action_dims, 0, action_dims, 2) = action_limits;
  }
  optimizer->setLimits(optimizer_limits);
  // Building reward function
  rosban_bbo::Optimizer::RewardFunc reward_function =
    [&p, &value_function, this, state, action_dims]
    (const Eigen::VectorXd & next_actions, std::default_random_engine * engine)
    {
      double total_reward = 0;
      for (int rollout = 0; rollout < this->rollouts_per_sample; rollout++) {
        double gain = 1.0;
        double rollout_reward = 0;
        bool is_terminated = false;
        Eigen::VectorXd curr_state = state;
        for (int step = 0; step < this->look_ahead; step++) {
          Eigen::VectorXd action(action_dims+1);
          action(0) = 0;
          action.segment(1,action_dims) = next_actions.segment(action_dims*step, action_dims);
          Problem::Result result = p.getSuccessor(curr_state, action, engine);
          rollout_reward += gain * result.reward;
          curr_state = result.successor;
          gain *= this->discount;
          is_terminated = result.terminal;
          // Stop predicting steps if a terminal state has been reached
          if (is_terminated) break;
        }
        if (!is_terminated) {
          rollout_reward += gain * value_function.predict(curr_state, 0);
        }
        total_reward += rollout_reward;
      }
      double avg_reward = total_reward / rollouts_per_sample;
      return avg_reward;
    };
  // Optimizing next actions
  Eigen::VectorXd next_actions = optimizer->train(reward_function, engine);
  // Only return next action
  return next_actions.segment(0,action_dims);
}

std::string OpenLoopPlanner::getClassName() const
{
  return "OpenLoopPlanner";
}

Json::Value OpenLoopPlanner::toJson() const
{
  Json::Value v;
  v["optimizer" ]          = optimizer->toFactoryJson();
  v["look_ahead"]          = look_ahead                ;
  v["rollouts_per_sample"] = rollouts_per_sample       ;
  v["discount"           ] = discount                  ;
  return v;
}

void OpenLoopPlanner::fromJson(const Json::Value & v,
                               const std::string & dir_name)
{
  rosban_bbo::OptimizerFactory().tryRead(v, "optimizer", dir_name, &optimizer);
  rhoban_utils::tryRead(v, "look_ahead"         , &look_ahead         );
  rhoban_utils::tryRead(v, "rollouts_per_sample", &rollouts_per_sample);
  rhoban_utils::tryRead(v, "discount"           , &discount           );
}



}
