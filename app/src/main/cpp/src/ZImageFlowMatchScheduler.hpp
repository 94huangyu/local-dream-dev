// Deterministic FlowMatch Euler scheduler used by the official Z-Image Turbo
// pipeline. This is deliberately separate from FlowMatchScheduler.hpp: that
// class implements Anima's ancestral/CONST sampler and sends sigma to its
// model, whereas Z-Image receives (1000 - scheduler_timestep) / 1000.
#ifndef ZIMAGEFLOWMATCHSCHEDULER_HPP
#define ZIMAGEFLOWMATCHSCHEDULER_HPP

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

#include "Scheduler.hpp"

class ZImageFlowMatchScheduler final : public Scheduler {
 public:
  explicit ZImageFlowMatchScheduler(float shift = 3.0f) : shift_(shift) {}

  // Mirrors ZImagePipeline.get_default_z_image_sigmas(): before shift, the
  // N values are linearly spaced from 1 to 1/N. Diffusers then applies the
  // static FlowMatch shift and appends terminal zero.
  void set_timesteps(int num_inference_steps) override {
    if (num_inference_steps <= 0)
      throw std::invalid_argument("Z-Image steps must be positive");

    std::vector<float> sigmas;
    sigmas.reserve(static_cast<size_t>(num_inference_steps) + 1);
    for (int i = 0; i < num_inference_steps; ++i) {
      const float raw =
          num_inference_steps == 1
              ? 1.0f
              : 1.0f - (1.0f - 1.0f / num_inference_steps) *
                           static_cast<float>(i) /
                           static_cast<float>(num_inference_steps - 1);
      sigmas.push_back(applyShift(raw));
    }

    std::vector<float> timesteps;
    timesteps.reserve(static_cast<size_t>(num_inference_steps));
    for (float sigma : sigmas) timesteps.push_back(sigma * kTrainTimesteps);
    sigmas.push_back(0.0f);

    sigmas_ = xt::adapt(std::move(sigmas));
    timesteps_ = xt::adapt(std::move(timesteps));
    step_index_.reset();
    begin_index_.reset();
  }

  xt::xarray<float> scale_model_input(const xt::xarray<float> &sample,
                                      int /*timestep*/) override {
    return sample;
  }

  // Diffusers FlowMatchEulerDiscreteScheduler's non-stochastic branch:
  // previous = sample + (sigma_next - sigma) * model_output.
  // The shared Scheduler interface truncates its timestep argument to int;
  // using the internally advanced index preserves the official float values.
  SchedulerOutput step(const xt::xarray<float> &model_output,
                       int /*timestep*/,
                       const xt::xarray<float> &sample) override {
    if (sigmas_.size() < 2)
      throw std::runtime_error("set_timesteps must be called before stepping");
    if (!step_index_) step_index_ = begin_index_.value_or(0);
    const size_t index = *step_index_;
    if (index + 1 >= sigmas_.size())
      throw std::runtime_error("Z-Image scheduler stepped past its schedule");

    const float dt = sigmas_(index + 1) - sigmas_(index);
    xt::xarray<float> previous = sample + dt * model_output;
    ++(*step_index_);
    return {previous, previous};
  }

  xt::xarray<float> add_noise(const xt::xarray<float> &original_samples,
                              const xt::xarray<float> &noise,
                              const xt::xarray<int> & /*timesteps*/) const override {
    const size_t index = std::min(begin_index_.value_or(0), sigmas_.size() - 1);
    const float sigma = sigmas_(index);
    return (1.0f - sigma) * original_samples + sigma * noise;
  }

  void set_begin_index(int begin_index) override {
    if (begin_index < 0) throw std::invalid_argument("begin index must be non-negative");
    begin_index_ = static_cast<size_t>(begin_index);
  }
  void set_prediction_type(const std::string &) override {}
  const xt::xarray<float> &get_timesteps() const override { return timesteps_; }
  size_t get_step_index() const override { return step_index_.value_or(0); }
  float get_current_sigma() const override {
    if (sigmas_.size() == 0) return 0.0f;
    return sigmas_(std::min(step_index_.value_or(0), sigmas_.size() - 1));
  }
  float get_init_noise_sigma() const override { return 1.0f; }

 private:
  static constexpr float kTrainTimesteps = 1000.0f;
  float applyShift(float sigma) const {
    return shift_ * sigma / (1.0f + (shift_ - 1.0f) * sigma);
  }

  float shift_;
  xt::xarray<float> sigmas_;
  xt::xarray<float> timesteps_;
  std::optional<size_t> step_index_;
  std::optional<size_t> begin_index_;
};

#endif  // ZIMAGEFLOWMATCHSCHEDULER_HPP
