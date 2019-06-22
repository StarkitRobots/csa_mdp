#pragma once

#include "starkit_csa_mdp/solvers/learner.h"

#include "starkit_utils/serialization/factory.h"

namespace csa_mdp
{
class LearnerFactory : public starkit_utils::Factory<Learner>
{
public:
  /// Automatically register several learners
  LearnerFactory();
};

}  // namespace csa_mdp
