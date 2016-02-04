#include "rosban_csa_mdp/solvers/fpf.h"
#include "rosban_csa_mdp/solvers/mre.h"

#include <ros/ros.h>

int main(int argc, char ** argv)
{
  std::string type = ros::getROSArg(argc, argv, "type");
  if (type == "")
  {
    std::cout << "Usage: ... type:=<type>" << std::endl;
  }

  Eigen::MatrixXd state_limits(3,2), action_limits(1,2);
  state_limits << -1, 1, -2, 2, -3, 3;
  action_limits << -1, 1;

  if (type == "FPF")
  {
    csa_mdp::FPF::Config conf;
    conf.setStateLimits(state_limits);
    conf.setActionLimits(action_limits);
    conf.pretty_print();
//    std::cout << conf.to_xml_stream() << std::endl;
  }
}
