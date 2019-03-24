#pragma once

#include "rhoban_csa_mdp/solvers/learner.h"

#include "rhoban_utils/serialization/factory.h"

namespace csa_mdp
{
class LearnerFactory : public rhoban_utils::Factory<Learner>
{
public:
  /// Automatically register several learners
  LearnerFactory();
};

}  // namespace csa_mdp
