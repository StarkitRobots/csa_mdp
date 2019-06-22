#include "starkit_csa_mdp/solvers/black_box_learner.h"

#include "starkit_csa_mdp/core/fa_policy.h"
#include "starkit_csa_mdp/core/problem_factory.h"

#include "starkit_random/tools.h"
#include "starkit_utils/threading/multi_core.h"

using starkit_utils::TimeStamp;

namespace csa_mdp
{
BlackBoxLearner::BlackBoxLearner()
  : nb_threads(1), time_budget(60), discount(0.98), trial_length(50), nb_evaluation_trials(100), iterations(0)
{
  openLogs();
}

BlackBoxLearner::~BlackBoxLearner()
{
  closeLogs();
}

std::unique_ptr<Policy> BlackBoxLearner::buildPolicy(const starkit_fa::FunctionApproximator& fa) const
{
  std::unique_ptr<Policy> p(new FAPolicy(fa.clone()));
  p->setActionLimits(problem->getActionsLimits());
  return p;
}

void BlackBoxLearner::run(std::default_random_engine* engine)
{
  init(engine);
  // Main learning loop
  learning_start = starkit_utils::TimeStamp::now();
  double elapsed = 0;
  while (elapsed < time_budget)
  {
    iterations++;
    std::cout << "Iteration " << iterations << std::endl;
    update(engine);
    // Stop if time has elapsed
    elapsed = diffSec(learning_start, starkit_utils::TimeStamp::now());
  }
}

double BlackBoxLearner::evaluatePolicy(const Policy& p, std::default_random_engine* engine) const
{
  return evaluatePolicy(p, nb_evaluation_trials, engine);
}

double BlackBoxLearner::evaluatePolicy(const Policy& p, int nb_evaluations, std::default_random_engine* engine,
                                       std::vector<Eigen::VectorXd>* visited_states) const
{
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = starkit_random::getRandomEngines(std::min(nb_threads, nb_evaluations), engine);
  // Rewards + visited_states are computed by different threads and stored in the same vector
  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(nb_evaluations);
  std::vector<std::vector<Eigen::VectorXd>> visited_states_per_thread(nb_evaluations);
  bool store_visited_states = visited_states != nullptr;
  // The task which has to be performed :
  starkit_utils::MultiCore::StochasticTask task = [this, &p, &rewards, &visited_states_per_thread, store_visited_states](
                                                     int start_idx, int end_idx, std::default_random_engine* engine) {
    for (int idx = start_idx; idx < end_idx; idx++)
    {
      Eigen::VectorXd state = problem->getStartingState(engine);
      double gain = 1.0;
      for (int step = 0; step < trial_length; step++)
      {
        if (store_visited_states)
        {
          visited_states_per_thread[idx].push_back(state);
        }
        Eigen::VectorXd action = p.getAction(state, engine);
        Problem::Result result = problem->getSuccessor(state, action, engine);
        double step_reward = result.reward;
        state = result.successor;
        rewards(idx) += gain * step_reward;
        gain = gain * discount;
        if (result.terminal)
          break;
      }
    }
  };
  // Running computation
  starkit_utils::MultiCore::runParallelStochasticTask(task, nb_evaluations, &engines);
  // Fill visited states if required
  if (store_visited_states)
  {
    for (const std::vector<Eigen::VectorXd>& eval_visited_states : visited_states_per_thread)
    {
      for (const Eigen::VectorXd& state : eval_visited_states)
      {
        visited_states->push_back(state);
      }
    }
  }
  // Result
  return rewards.mean();
}

double BlackBoxLearner::localEvaluation(const Policy& p, const Eigen::MatrixXd& space, int nb_evaluations,
                                        std::default_random_engine* engine) const
{
  // Sampling starting states
  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(nb_evaluations);
  // The task which has to be performed :
  starkit_utils::MultiCore::StochasticTask task = [this, &p, &rewards, &space](int start_idx, int end_idx,
                                                                              std::default_random_engine* engine) {
    // 1: Generating states
    int thread_evaluations = end_idx - start_idx;
    std::vector<Eigen::VectorXd> starting_states;
    starting_states = starkit_random::getUniformSamples(space, thread_evaluations, engine);
    // 2: Simulating trajectories
    try
    {
      for (int idx = 0; idx < thread_evaluations; idx++)
      {
        Eigen::VectorXd state = starting_states[idx];
        double gain = 1.0;
        for (int step = 0; step < trial_length; step++)
        {
          Eigen::VectorXd action = p.getAction(state, engine);
          Problem::Result result = problem->getSuccessor(state, action, engine);
          state = result.successor;
          rewards(idx + start_idx) += gain * result.reward;
          gain = gain * discount;
          if (result.terminal)
            break;
        }
      }
    }
    catch (const std::runtime_error& exc)
    {
      std::ostringstream oss;
      oss << "BlackBoxLearner::localEvaluation:task: " << exc.what() << std::endl;
      std::cerr << oss.str();
      throw exc;
    }
  };
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = starkit_random::getRandomEngines(std::min(nb_threads, nb_evaluations), engine);
  // Running computation
  starkit_utils::MultiCore::runParallelStochasticTask(task, nb_evaluations, &engines);
  // Result
  return rewards.mean();
}

double BlackBoxLearner::evaluation(const Policy& p, const std::vector<Eigen::VectorXd>& initial_states,
                                   std::default_random_engine* engine) const
{
  // Initializing reward list
  int nb_evaluations = initial_states.size();
  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(nb_evaluations);
  // The task which has to be performed :
  starkit_utils::MultiCore::StochasticTask task =
      [this, &p, &rewards, &initial_states](int start_idx, int end_idx, std::default_random_engine* engine) {
        // Simulating trajectories
        for (int idx = start_idx; idx < end_idx; idx++)
        {
          Eigen::VectorXd state = initial_states[idx];
          double gain = 1.0;
          for (int step = 0; step < trial_length; step++)
          {
            Eigen::VectorXd action = p.getAction(state, engine);
            Problem::Result result = problem->getSuccessor(state, action, engine);
            state = result.successor;
            rewards(idx) += gain * result.reward;
            gain = gain * discount;
            if (result.terminal)
              break;
          }
        }
      };
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = starkit_random::getRandomEngines(std::min(nb_threads, nb_evaluations), engine);
  // Running computation
  starkit_utils::MultiCore::runParallelStochasticTask(task, nb_evaluations, &engines);
  // Result
  return rewards.mean();
}

void BlackBoxLearner::setNbThreads(int nb_threads_)
{
  nb_threads = nb_threads_;
}

Json::Value BlackBoxLearner::toJson() const
{
  throw std::logic_error("BlackBoxLearner::toJson: not implemented");
}

void BlackBoxLearner::fromJson(const Json::Value& v, const std::string& dir_name)
{
  // Reading simple parameters
  starkit_utils::tryRead(v, "nb_threads", &nb_threads);
  starkit_utils::tryRead(v, "trial_length", &trial_length);
  starkit_utils::tryRead(v, "nb_evaluation_trials", &nb_evaluation_trials);
  starkit_utils::tryRead(v, "verbosity", &verbosity);
  starkit_utils::tryRead(v, "time_budget", &time_budget);
  starkit_utils::tryRead(v, "discount", &discount);

  // Getting problem
  std::shared_ptr<const Problem> tmp_problem;
  std::string problem_path;
  starkit_utils::tryRead(v, "problem_path", &problem_path);
  if (problem_path != "")
  {
    tmp_problem = ProblemFactory().buildFromJsonFile(dir_name + problem_path);
  }
  else
  {
    tmp_problem = ProblemFactory().read(v, "problem", dir_name);
  }
  problem = std::dynamic_pointer_cast<const BlackBoxProblem>(tmp_problem);
  if (!problem)
  {
    throw std::runtime_error("BlackBoxLearner::fromJson: problem is not a BlackBoxProblem");
  }
}

void BlackBoxLearner::openLogs()
{
  // Opening files
  time_file.open("time.csv");
  results_file.open("results.csv");
  // Writing headers
  time_file << "iteration,part,time" << std::endl;
  results_file << "iteration,score,elapsed" << std::endl;
}

void BlackBoxLearner::closeLogs()
{
  time_file.close();
  results_file.close();
}

void BlackBoxLearner::writeTime(const std::string& name, double time)
{
  // TODO:
  // - Add members once available (require implementation in AdaptativeTree)
  time_file << iterations << "," << name << "," << time << std::endl;
}

void BlackBoxLearner::writeScore(double score)
{
  double elapsed = diffSec(learning_start, starkit_utils::TimeStamp::now());
  results_file << iterations << "," << score << "," << elapsed << std::endl;
}

}  // namespace csa_mdp
