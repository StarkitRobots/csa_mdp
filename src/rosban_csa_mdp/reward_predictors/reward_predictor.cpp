#include "rosban_csa_mdp/reward_predictors/reward_predictor.h"

namespace csa_mdp
{

RewardPredictor::RewardPredictor() : nb_threads(1) {}
RewardPredictor::~RewardPredictor() {}

void RewardPredictor::setNbThreads(int new_nb_threads)
{
  nb_threads = new_nb_threads;
}

void RewardPredictor::to_xml(std::ostream & out) const
{
  rosban_utils::xml_tools::write<int>("nb_threads", nb_threads, out);
}

void RewardPredictor::from_xml(TiXmlNode * node)
{
  rosban_utils::xml_tools::try_read<int>(node, "nb_threads", nb_threads);
}

}
