#pragma once

#include "rhoban_csa_mdp/action_optimizers/action_optimizer.h"

#include "rhoban_fa/trainer.h"

namespace csa_mdp
{

class BasicOptimizer : public ActionOptimizer
{
public:
  BasicOptimizer();

  virtual Eigen::VectorXd optimize(const Eigen::VectorXd & input,
                                   const Eigen::MatrixXd & action_limits,
                                   std::shared_ptr<const Policy> current_policy,
                                   Problem::ResultFunction result_function,
                                   Problem::ValueFunction value_function,
                                   double discount,
                                   std::default_random_engine * engine) const override;

  typedef std::function<void(int start, int end, std::default_random_engine * engine)> AOTask;

  virtual AOTask getTask(const Eigen::VectorXd & input,
                         const Eigen::MatrixXd & actions,
                         std::shared_ptr<const Policy> policy,
                         Problem::ResultFunction result_function,
                         Problem::ValueFunction value_function,
                         double discount,
                         Eigen::VectorXd & results) const;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

private:
  int nb_additional_steps;
  int nb_simulations;
  int nb_actions;
  std::unique_ptr<rhoban_fa::Trainer> trainer;
};

}
