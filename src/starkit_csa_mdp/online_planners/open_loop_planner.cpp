#include "starkit_csa_mdp/online_planners/open_loop_planner.h"

#include "starkit_bbo/optimizer_factory.h"

namespace csa_mdp
{
OpenLoopPlanner::OpenLoopPlanner() : look_ahead(0), rollouts_per_sample(1), discount(1)
{
}

void OpenLoopPlanner::checkConsistency(const Problem& p) const
{
  if (p.getNbActions() != 1)
  {
    throw std::runtime_error("OpenLoopPlanner::checkConsistency: no support for hybrid action spaces");
  }
  if (look_ahead <= 0)
  {
    throw std::runtime_error("OpenLoopPlanner::checkConsistency: invalid value for look_ahead: '" +
                             std::to_string(look_ahead) + "'");
  }
}

void OpenLoopPlanner::prepareOptimizer(const Problem& p)
{
  Eigen::MatrixXd action_limits = p.getActionLimits(0);
  int action_dims = action_limits.rows();
  Eigen::MatrixXd optimizer_limits(action_dims * look_ahead, 2);
  for (int action = 0; action < look_ahead; action++)
  {
    optimizer_limits.block(action * action_dims, 0, action_dims, 2) = action_limits;
  }
  optimizer->setLimits(optimizer_limits);
}

double OpenLoopPlanner::sampleLookAheadReward(const Problem& p, const Eigen::VectorXd& initial_state,
                                              const Eigen::VectorXd& next_actions, Eigen::VectorXd* last_state,
                                              bool* is_terminated, std::default_random_engine* engine)
{
  int action_dims = p.actionDims(0);
  double gain = 1.0;
  double rollout_reward = 0;
  Eigen::VectorXd curr_state = initial_state;
  for (int step = 0; step < look_ahead; step++)
  {
    Eigen::VectorXd action(action_dims + 1);
    action(0) = 0;
    action.segment(1, action_dims) = next_actions.segment(action_dims * step, action_dims);
    Problem::Result result = p.getSuccessor(curr_state, action, engine);
    rollout_reward += gain * result.reward;
    curr_state = result.successor;
    gain *= discount;
    // Stop predicting steps if a terminal state has been reached
    if (result.terminal)
    {
      *is_terminated = true;
      *last_state = curr_state;
      break;
    }
  }
  *last_state = curr_state;
  return rollout_reward;
}

Eigen::VectorXd OpenLoopPlanner::planNextAction(const Problem& p, const Eigen::VectorXd& state, const Policy& policy,
                                                std::default_random_engine* engine)
{
  checkConsistency(p);
  prepareOptimizer(p);
  // Building reward function
  starkit_bbo::Optimizer::RewardFunc reward_function = [&p, &policy, this, state](const Eigen::VectorXd& next_actions,
                                                                                 std::default_random_engine* engine) {
    double total_reward = 0;
    for (int rollout = 0; rollout < this->rollouts_per_sample; rollout++)
    {
      bool trial_terminated = false;
      Eigen::VectorXd final_state;
      double rollout_reward =
          this->sampleLookAheadReward(p, state, next_actions, &final_state, &trial_terminated, engine);
      // If rollout has not ended with a terminal status, use policy to end
      // the trial
      if (!trial_terminated)
      {
        double future_reward =
            p.sampleRolloutReward(final_state, policy, trial_length - look_ahead, this->discount, engine);
        rollout_reward += future_reward * pow(this->discount, look_ahead);
      }
      total_reward += rollout_reward;
    }
    double avg_reward = total_reward / rollouts_per_sample;
    return avg_reward;
  };
  // Optimizing next actions
  Eigen::VectorXd next_actions = optimizer->train(reward_function, engine);
  // Only return next action, with a prefix
  int action_dims = p.actionDims(0);
  Eigen::VectorXd prefixed_action(1 + action_dims);
  prefixed_action(0) = 0;
  prefixed_action.segment(1, action_dims) = next_actions;
  return prefixed_action;
}

Eigen::VectorXd OpenLoopPlanner::planNextAction(const Problem& p, const Eigen::VectorXd& state,
                                                const starkit_fa::FunctionApproximator& value_function,
                                                std::default_random_engine* engine)
{
  checkConsistency(p);
  prepareOptimizer(p);
  // Building reward function
  starkit_bbo::Optimizer::RewardFunc reward_function =
      [&p, &value_function, this, state](const Eigen::VectorXd& next_actions, std::default_random_engine* engine) {
        double total_reward = 0;
        for (int rollout = 0; rollout < this->rollouts_per_sample; rollout++)
        {
          bool trial_terminated = false;
          Eigen::VectorXd final_state;
          double rollout_reward =
              this->sampleLookAheadReward(p, state, next_actions, &final_state, &trial_terminated, engine);
          // If rollout has not ended with a terminal status, use policy to end
          // the trial
          if (!trial_terminated)
          {
            rollout_reward += pow(this->discount, look_ahead) * value_function.predict(final_state, 0);
          }
          total_reward += rollout_reward;
        }
        double avg_reward = total_reward / rollouts_per_sample;
        return avg_reward;
      };
  // Optimizing next actions
  Eigen::VectorXd next_actions = optimizer->train(reward_function, engine);
  // Only return next action, with a prefix
  int action_dims = p.actionDims(0);
  Eigen::VectorXd prefixed_action(1 + action_dims);
  prefixed_action(0) = 0;
  prefixed_action.segment(1, action_dims) = next_actions.segment(0, action_dims);
  return prefixed_action;
}

std::string OpenLoopPlanner::getClassName() const
{
  return "OpenLoopPlanner";
}

Json::Value OpenLoopPlanner::toJson() const
{
  Json::Value v;
  v["optimizer"] = optimizer->toFactoryJson();
  v["look_ahead"] = look_ahead;
  v["trial_length"] = trial_length;
  v["rollouts_per_sample"] = rollouts_per_sample;
  v["discount"] = discount;
  return v;
}

void OpenLoopPlanner::fromJson(const Json::Value& v, const std::string& dir_name)
{
  starkit_bbo::OptimizerFactory().tryRead(v, "optimizer", dir_name, &optimizer);
  starkit_utils::tryRead(v, "look_ahead", &look_ahead);
  starkit_utils::tryRead(v, "trial_length", &trial_length);
  starkit_utils::tryRead(v, "rollouts_per_sample", &rollouts_per_sample);
  starkit_utils::tryRead(v, "discount", &discount);
}

}  // namespace csa_mdp
