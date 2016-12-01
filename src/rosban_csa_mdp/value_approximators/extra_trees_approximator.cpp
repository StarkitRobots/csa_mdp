#include "rosban_csa_mdp/value_approximators/extra_trees_approximator.h"

#include "rosban_csa_mdp/reward_predictors/reward_predictor_factory.h"
#include "rosban_fa/trainer_factory.h"

#include "rosban_random/tools.h"

#include "rosban_utils/multi_core.h"

using rosban_utils::MultiCore;

namespace csa_mdp
{

ExtraTreesApproximator::ExtraTreesApproximator()
  : nb_samples(100) {}

ExtraTreesApproximator::~ExtraTreesApproximator() {}

std::unique_ptr<rosban_fa::FunctionApproximator>
ExtraTreesApproximator::train(const Policy & policy,
                              const Problem & problem,
                              Problem::ValueFunction current_value_function,
                              double discount,
                              std::default_random_engine * engine)
{
  if (!predictor) {
    throw std::logic_error("ExtraTreesApproximator::train: predictor is not initialized");
  }
  if (!trainer) {
    throw std::logic_error("ExtraTreesApproximator::train: trainer is not initialized");
  }
  Eigen::MatrixXd inputs =
    rosban_random::getUniformSamplesMatrix(problem.getStateLimits(),
                                           nb_samples,
                                           engine);
  Eigen::VectorXd observations(nb_samples);
  // Creating reward predictor task
  MultiCore::StochasticTask rp_task;
  rp_task = [this, &policy, &problem, discount, current_value_function, &inputs, &observations]
    (int start_idx, int end_idx, std::default_random_engine * thread_engine)
    {
      for (int sample = start_idx; sample < end_idx; sample++)
      {
        Eigen::VectorXd state = inputs.col(sample);
        double mean, var;
        predictor->predict(state, policy,
                           problem.getTransitionFunction(),
                           problem.getRewardFunction(),
                           current_value_function,
                           problem.getTerminalFunction(),
                           discount,
                           &mean, &var,
                           thread_engine);
        observations(sample) = mean;
      }
    };
  // Choosing how many threads will be used for the samples and how many
  // subthreads will be spawned by each thread
  int wished_threads = nb_threads;
  int subthreads = 1;
  // Increasing performances when the number of samples is small, but ensuring that
  // no more than nb_threads are created
  if (nb_threads > nb_samples) {
    subthreads = std::ceil(nb_threads / (double)nb_samples);
    wished_threads = nb_threads / subthreads;
  }
  predictor->setNbThreads(subthreads);

  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = rosban_random::getRandomEngines(wished_threads, engine);
  // Run threads in parallel
  MultiCore::runParallelStochasticTask(rp_task, nb_samples, &engines);
  // Approximate the gathered samples
  return trainer->train(inputs, observations, problem.getStateLimits());
}

void ExtraTreesApproximator::setNbThreads(int nb_threads_) {
  nb_threads = nb_threads_;
  if (predictor)
    predictor->setNbThreads(nb_threads);
  if (trainer)
    trainer->setNbThreads(nb_threads);
}

std::string ExtraTreesApproximator::class_name() const
{
  return "ExtraTreesApproximator";
}

void ExtraTreesApproximator::to_xml(std::ostream &out) const {
  ValueApproximator::to_xml(out);
}

void ExtraTreesApproximator::from_xml(TiXmlNode *node) {
  ValueApproximator::from_xml(node);
  RewardPredictorFactory().tryRead   (node, "predictor", predictor);
  rosban_fa::TrainerFactory().tryRead(node, "trainer"  , trainer  );
}


}
