#pragma once
// Fixed-timestep accumulator (the canonical "Fix Your Timestep" pattern). Backend-agnostic, pure
// (hf_core): no window, no clock source of its own — the caller feeds it the real frame dt and it
// reports how many fixed simulation steps to run, carrying the remainder. Deterministic given the
// same dt sequence, so it is fully unit-testable without a window.

namespace hf::runtime {

class FixedTimestep {
public:
    // stepSeconds: the fixed simulation step (default 1/120 s).
    // maxStepsPerTick: cap on steps run for one Tick to avoid the "spiral of death" when a frame
    //                  hitches badly; excess accumulated time beyond the cap is discarded.
    explicit FixedTimestep(float stepSeconds = 1.0f / 120.0f, int maxStepsPerTick = 8)
        : step_(stepSeconds > 0.0f ? stepSeconds : 1.0f / 120.0f),
          maxSteps_(maxStepsPerTick > 0 ? maxStepsPerTick : 1) {}

    // Accumulate realDt and return how many fixed steps to run this tick. Leaves the leftover in the
    // accumulator (< step_, unless the cap discarded the excess). Negative dt is ignored.
    int Tick(float realDtSeconds) {
        if (realDtSeconds > 0.0f) accumulator_ += realDtSeconds;
        int steps = 0;
        while (accumulator_ >= step_ && steps < maxSteps_) {
            accumulator_ -= step_;
            ++steps;
        }
        // Cap hit: drop the backlog so we don't perpetually fall behind (spiral-of-death guard).
        if (steps == maxSteps_ && accumulator_ >= step_) {
            accumulator_ = 0.0f;
        }
        return steps;
    }

    float Step() const { return step_; }
    float Accumulator() const { return accumulator_; }
    // Interpolation fraction (0..1) of the way into the next fixed step, for render interpolation.
    float Alpha() const { return accumulator_ / step_; }
    void Reset() { accumulator_ = 0.0f; }

private:
    float step_;
    int   maxSteps_;
    float accumulator_ = 0.0f;
};

} // namespace hf::runtime
