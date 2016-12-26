#include "rosban_csa_mdp/solvers/black_box_learner.h"

#include "rosban_csa_mdp/core/problem_factory.h"

#include "rosban_random/tools.h"
#include "rosban_utils/multi_core.h"

using rosban_utils::TimeStamp;

namespace csa_mdp
{

BlackBoxLearner::BlackBoxLearner()
  : nb_threads(1),
    time_budget(60),
    discount(0.98),
    trial_length(50),
    nb_evaluation_trials(100),
    iterations(0)
{
  openLogs();
}

BlackBoxLearner::~BlackBoxLearner()
{
  closeLogs();
}

void BlackBoxLearner::run(std::default_random_engine * engine)
{
  init(engine);
  // Main learning loop
  learning_start = rosban_utils::TimeStamp::now();
  double elapsed = 0;
  while (elapsed < time_budget) {
    iterations++;
    std::cout << "Iteration " << iterations << std::endl;
    update(engine);
    // Stop if time has elapsed
    elapsed = diffSec(learning_start, rosban_utils::TimeStamp::now());
  }
}


double BlackBoxLearner::evaluatePolicy(const Policy & p,
                                       std::default_random_engine * engine) const
{
  // Rewards are computed by different threads and stored in the same vector
  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(nb_evaluation_trials);
  // The task which has to be performed :
  rosban_utils::MultiCore::StochasticTask task =
    [this, &p, &rewards]
    (int start_idx, int end_idx, std::default_random_engine * engine)
    {
      for (int idx = start_idx; idx < end_idx; idx++) {
        Eigen::VectorXd state = problem->getStartingState(engine);
        double gain = 1.0;
        for (int step = 0; step < trial_length; step++) {
          Eigen::VectorXd action = p.getAction(state, engine);
          Eigen::VectorXd next_state = problem->getSuccessor(state, action, engine);
          double step_reward = problem->getReward(state, action, next_state);
          state = next_state;
          rewards(idx) += gain * step_reward;
          gain = gain * discount;
          if (problem->isTerminal(state)) break;
        }
      }
    };
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = rosban_random::getRandomEngines(std::min(nb_threads, nb_evaluation_trials), engine);
  // Running computation
  rosban_utils::MultiCore::runParallelStochasticTask(task, nb_evaluation_trials, &engines);
  // Result
  return rewards.mean();
}

double BlackBoxLearner::localEvaluation(const Policy & p,
                                        const Eigen::MatrixXd & space,
                                        int nb_evaluations,
                                        std::default_random_engine * engine) const
{
  // Sampling starting states
  std::vector<Eigen::VectorXd> starting_states =
    rosban_random::getUniformSamples(space, nb_evaluations, engine);
  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(nb_evaluations);
  // The task which has to be performed :
  rosban_utils::MultiCore::StochasticTask task =
    [this, &p, &starting_states, &rewards]
    (int start_idx, int end_idx, std::default_random_engine * engine)
    {
      for (int idx = start_idx; idx < end_idx; idx++) {
        Eigen::VectorXd state = starting_states[idx];
        double gain = 1.0;
        for (int step = 0; step < trial_length; step++) {
          Eigen::VectorXd action = p.getAction(state, engine);
          Eigen::VectorXd next_state = problem->getSuccessor(state, action, engine);
          double step_reward = problem->getReward(state, action, next_state);
          state = next_state;
          rewards(idx) += gain * step_reward;
          gain = gain * discount;
          if (problem->isTerminal(state)) break;
        }
      }
    };
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = rosban_random::getRandomEngines(std::min(nb_threads, nb_evaluations), engine);
  // Running computation
  rosban_utils::MultiCore::runParallelStochasticTask(task, nb_evaluations, &engines);
  // Result
  return rewards.mean();
}

void BlackBoxLearner::setNbThreads(int nb_threads_)
{
  nb_threads = nb_threads_;
}

void BlackBoxLearner::to_xml(std::ostream &out) const
{
  //TODO
  (void) out;
  throw std::logic_error("BlackBoxLearner::to_xml: not implemented");
}

void BlackBoxLearner::from_xml(TiXmlNode *node)
{
  // Reading simple parameters
  rosban_utils::xml_tools::try_read<int>   (node, "nb_threads"          , nb_threads          );
  rosban_utils::xml_tools::try_read<int>   (node, "trial_length"        , trial_length        );
  rosban_utils::xml_tools::try_read<int>   (node, "nb_evaluation_trials", nb_evaluation_trials);
  rosban_utils::xml_tools::try_read<int>   (node, "verbosity"           , verbosity           );
  rosban_utils::xml_tools::try_read<double>(node, "time_budget"         , time_budget         );
  rosban_utils::xml_tools::try_read<double>(node, "discount"            , discount            );

  // Getting problem
  std::shared_ptr<const Problem> tmp_problem;
  tmp_problem = ProblemFactory().read(node, "problem");
  problem = std::dynamic_pointer_cast<const BlackBoxProblem>(tmp_problem);
  if (!problem) {
    throw std::runtime_error("BlackBoxLearner::from_xml: problem is not a BlackBoxProblem");
  }
}

void BlackBoxLearner::openLogs()
{
  // Opening files
  time_file.open("time.csv");
  results_file.open("results.csv");
  // Writing headers
  time_file << "iteration,part,time" << std::endl;
  results_file << "iteration,score" << std::endl;
}


void BlackBoxLearner::closeLogs()
{
  time_file.close();
  results_file.close();
}

void BlackBoxLearner::writeTime(const std::string & name, double time)
{
  //TODO:
  // - Add members once available (require implementation in AdaptativeTree)
  time_file << iterations << "," << name << "," << time << std::endl;
}

void BlackBoxLearner::writeScore(double score) {
  results_file << iterations << "," << score << std::endl;
}


}
