#pragma once

#include "rosban_csa_mdp/core/problem.h"

#include "rhoban_utils/serialization/json_serializable.h"
#include "rosban_fa/function_approximator.h"
#include "rosban_bbo/optimizer.h"


namespace csa_mdp
{

class OpenLoopPlanner : public rhoban_utils::JsonSerializable
{
public:
  OpenLoopPlanner();

  /// Optimize the next action for the given problem 'p' starting at 'state'
  /// according to inner parameters and provided value_function
  Eigen::VectorXd planNextAction(const Problem & p,
                                 const Eigen::VectorXd & state,
                                 const rosban_fa::FunctionApproximator & value_function,
                                 std::default_random_engine * engine);

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;


private:
  /// Optimizer used for open loop planning
  std::unique_ptr<rosban_bbo::Optimizer> optimizer;

  /// Number of steps of look ahead
  int look_ahead;

  /// Number of rollouts used to average the reward at each sample
  int rollouts_per_sample;

  /// Discount value used for optimization
  double discount;
};

}
