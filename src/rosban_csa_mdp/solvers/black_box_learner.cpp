#include "rosban_csa_mdp/solvers/black_box_learner.h"

namespace csa_mdp
{

BlackBoxLearner::BlackBoxLearner()
  : nb_threads(1),
    discount(0.98),
    trial_length(50),
    nb_evaluation_trials(100),
    best_score(std::numeric_limits<double>::lowest())
{
}

BlackBoxLearner::~BlackBoxLearner() {}

void BlackBoxLearner::run()
{
  learning_start = rosban_utils::TimeStamp::now();
  int iteration = 0;
  while (true) {
    // Update function approximatiors
    updateValue();
    updatePolicy();
    // Stop if time has elapsed
    double elapsed = diffSec(learning_start, rosban_utils::TimeStamp::now());
    if (elapsed > allowed_time)
      break;
    // evaluate and save policy if it's better than previous ones
    double score = evaluatePolicy();
    if (score > best_score) {
      best_score = score;
      //TODO save policy (issues yet)
    }
    std::cout << iteration << ": " << score << std::endl;
    iteration++;
  }
}

double BlackBoxLearner::evaluatePolicy()
{
  double reward = 0;
  for (int trial = 0; trial < nb_evaluation_trials; trial++) {
    Eigen::VectorXd state = problem->getStartingState();
    double gain = 1.0;
    for (int step = 0; step < trial_length; step++) {
      Eigen::VectorXd action = policy->getAction(state);
      Sample s = problem->getSample(state, action);
      state = s.next_state;
      reward += gain * s.reward;
      gain = gain * discount;
    }
  }
  return reward / nb_evaluation_trials;
}

void BlackBoxLearner::setNbThreads(int nb_threads)
{
  //TODO
}

void BlackBoxLearner::to_xml(std::ostream &out) const
{
  //TODO
}

void BlackBoxLearner::from_xml(TiXmlNode *node)
{
  //TODO
}


}
