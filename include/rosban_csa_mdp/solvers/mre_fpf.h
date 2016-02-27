#pragma once

#include "rosban_csa_mdp/solvers/fpf.h"
#include "rosban_csa_mdp/knownness/knownness_function.h"

namespace csa_mdp
{

/// A class implementing the Multi-Resolution Exploration adapted on FPF
class MREFPF : public FPF
{
public:
  /// Two types of KnownnessTrees are available
  /// MRE: Follows Multi-Resolution Exploration description (Nouri2009)
  ///      i.e. samples are modified to include knownness function
  /// Alternative: The trees are modified to apply the knownness function
  enum class UpdateType
  { MRE, Alternative };

  class Config : public FPF::Config
  {
  public:

    Config();

    // XML stuff
    virtual std::string class_name() const override;
    virtual void to_xml(std::ostream &out) const override;
    virtual void from_xml(TiXmlNode *node) override;

    bool filter_samples;
    double reward_max;
    UpdateType update_type;
  };

  MREFPF();
  MREFPF(const Config & conf,
         std::shared_ptr<KnownnessFunction> knownness_func);

protected:
  virtual regression_forests::TrainingSet
  getTrainingSet(const std::vector<Sample>& samples,
                 std::function<bool(const Eigen::VectorXd&)> is_terminal) override;

  virtual void
  updateQValue(const std::vector<Sample> &samples,
               std::function<bool(const Eigen::VectorXd&)> is_terminal) override;

  std::vector<Sample> filterSimilarSamples(const std::vector<Sample> &samples) const;

public:
  Config conf;
private:
  std::shared_ptr<KnownnessFunction> knownness_func;
};

std::string to_string(MREFPF::UpdateType type);
MREFPF::UpdateType loadUpdateType(const std::string &type);

}
