#include "rosban_csa_mdp/solvers/fpf.h"
#include "rosban_csa_mdp/solvers/mre.h"

int main(int argc, char ** argv)
{
  if (argc < 2)
  {
    std::cout << "Usage: ... <type>" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string type(argv[1]);

  Eigen::MatrixXd state_limits(3,2), action_limits(1,2);
  state_limits << -1, 1, -2, 2, -3, 3;
  action_limits << -1, 1;

  if (type == "FPF")
  {
    csa_mdp::FPF::Config conf;
    conf.setStateLimits(state_limits);
    conf.setActionLimits(action_limits);
    conf.pretty_print();
  }
  else
  {
    throw std::runtime_error("Unknown type '" + type + "'");
  }
}
