#include "rhoban_csa_mdp/value_approximators/extra_trees_approximator.h"

#include "rhoban_csa_mdp/reward_predictors/reward_predictor_factory.h"
#include "rhoban_fa/trainer_factory.h"

#include "rhoban_random/tools.h"

#include "rhoban_utils/threading/multi_core.h"
#include "rhoban_utils/io_tools.h"

using rhoban_utils::MultiCore;

namespace csa_mdp
{

ExtraTreesApproximator::ExtraTreesApproximator()
  : nb_samples(100) {}

ExtraTreesApproximator::~ExtraTreesApproximator() {}

std::unique_ptr<rhoban_fa::FunctionApproximator>
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
    rhoban_random::getUniformSamplesMatrix(problem.getStateLimits(),
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
                           problem.getResultFunction(),
                           current_value_function,
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
  engines = rhoban_random::getRandomEngines(wished_threads, engine);
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

std::string ExtraTreesApproximator::getClassName() const
{
  return "ExtraTreesApproximator";
}

Json::Value ExtraTreesApproximator::toJson() const {
  Json::Value v = ValueApproximator::toJson();
  v["nb_samples"] = nb_samples;
  v["predictor"] = predictor->toFactoryJson();
  v["trainer"] = trainer->toFactoryJson();
  return v;
}

void ExtraTreesApproximator::fromJson(const Json::Value & v, const std::string & dir_name)
{
  ValueApproximator::fromJson(v, dir_name);
  rhoban_utils::tryRead(v, "nb_samples", &nb_samples);
  RewardPredictorFactory().tryRead   (v, "predictor", dir_name, &predictor);
  rhoban_fa::TrainerFactory().tryRead(v, "trainer"  , dir_name, &trainer  );
  setNbThreads(nb_threads);
}


}
