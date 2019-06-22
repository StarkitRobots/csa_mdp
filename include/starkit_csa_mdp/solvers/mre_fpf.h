#pragma once

#include "starkit_csa_mdp/solvers/fpf.h"
#include "starkit_csa_mdp/knownness/knownness_function.h"

namespace csa_mdp
{
/// A class implementing the Multi-Resolution Exploration adapted on FPF
class MREFPF : public FPF
{
public:
  /// Four types of use of the KnownnessTrees are available
  /// MRE        : Follows Multi-Resolution Exploration description (Nouri2009)
  ///              i.e. samples are modified to include knownness function
  /// Alternative: The trees are modified to apply the knownness function
  /// Disabled   : Values of the knownness trees are simply ignored
  /// MRE_CI     : Is also based on noise estimation, to avoid trying to explore
  ///              areas were the upper confidence bound is still low
  enum class UpdateType
  {
    MRE,
    Alternative,
    Disabled,
    MRE_CI
  };

  class Config : public FPF::Config
  {
  public:
    Config();

    // XML stuff
    virtual std::string getClassName() const override;
    virtual Json::Value toJson() const override;
    virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

    bool filter_samples;
    double reward_max;
    UpdateType update_type;
  };

  MREFPF();
  MREFPF(std::shared_ptr<KnownnessFunction> knownness_func);

  void setKnownnessFunc(std::shared_ptr<KnownnessFunction> knownness_func);

protected:
  /// TrueType of conf must be MREFPF::Config
  virtual regression_forests::TrainingSet getTrainingSet(const std::vector<Sample>& samples,
                                                         std::function<bool(const Eigen::VectorXd&)> is_terminal,
                                                         const FPF::Config& conf, int start_idx, int end_idx) override;

  /// TrueType of conf must be MREFPF::Config
  virtual void updateQValue(const std::vector<Sample>& samples, std::function<bool(const Eigen::VectorXd&)> is_terminal,
                            FPF::Config& conf, bool last_step) override;

  std::vector<Sample> filterSimilarSamples(const std::vector<Sample>& samples) const;

private:
  std::shared_ptr<KnownnessFunction> knownness_func;
};

std::string to_string(MREFPF::UpdateType type);
MREFPF::UpdateType loadUpdateType(const std::string& type);

}  // namespace csa_mdp
