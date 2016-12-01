#include "rosban_csa_mdp/value_approximators/value_approximator.h"

#include "rosban_utils/xml_tools.h"

namespace csa_mdp
{

ValueApproximator::ValueApproximator()
  : nb_threads(1) {}

ValueApproximator::~ValueApproximator() {}

void ValueApproximator::setNbThreads(int nb_threads_) {
  nb_threads = nb_threads_;
}

void ValueApproximator::to_xml(std::ostream &out) const {
  rosban_utils::xml_tools::write("nb_threads", nb_threads, out);
}

void ValueApproximator::from_xml(TiXmlNode *node) {
  rosban_utils::xml_tools::try_read(node, "nb_threads", nb_threads);
}


}
