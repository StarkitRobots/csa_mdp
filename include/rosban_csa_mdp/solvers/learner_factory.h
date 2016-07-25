#pragma once

#include "rosban_csa_mdp/solvers/learner.h"

#include "rosban_utils/factory.h"

namespace csa_mdp
{

class LearnerFactory : public rosban_utils::Factory<Learner>
{
public:
  /// Automatically register several learners
  LearnerFactory();
};

}
