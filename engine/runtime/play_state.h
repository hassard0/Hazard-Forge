#pragma once
// Slice AB — editor play/pause/step run-state. Pure C++ (hf_core), header-only, window-free.
//
// Wraps the simulation run-state so the fixed-timestep loop can freeze/advance the world from the
// editor: when Paused the fixed clock keeps accumulating real time (so resuming is seamless) but no
// simulation steps are consumed, EXCEPT a single-step request advances exactly one step. The loop
// calls StepsThisTick(fixedSteps) with the count FixedTimestep::Tick returned and uses the result as
// the number of steps to actually simulate. Trivially unit-testable (see tests/editor_test.cpp).

namespace hf::runtime {

enum class RunState { Playing, Paused };

class PlayState {
public:
    PlayState() = default;
    explicit PlayState(RunState s) : state_(s) {}

    RunState State() const { return state_; }
    bool IsPlaying() const { return state_ == RunState::Playing; }
    bool IsPaused() const { return state_ == RunState::Paused; }

    void Play()  { state_ = RunState::Playing; }
    void Pause() { state_ = RunState::Paused; }
    void Toggle() { state_ = IsPlaying() ? RunState::Paused : RunState::Playing; }

    // Request a single fixed step while paused. Consumed by the next StepsThisTick.
    void RequestStep() { stepRequested_ = true; }
    bool StepPending() const { return stepRequested_; }

    // Given the number of fixed steps the clock accumulated this tick, return how many to actually
    // simulate: Playing -> all of them; Paused -> 0, unless a single step was requested -> exactly 1
    // (the request is consumed regardless of fixedSteps, so Step works even on a frame the clock
    // produced 0 accumulated steps).
    int StepsThisTick(int fixedSteps) {
        if (state_ == RunState::Playing) {
            stepRequested_ = false;  // a queued step is irrelevant while playing
            return fixedSteps > 0 ? fixedSteps : 0;
        }
        if (stepRequested_) {
            stepRequested_ = false;
            return 1;
        }
        return 0;
    }

private:
    RunState state_ = RunState::Playing;
    bool stepRequested_ = false;
};

}  // namespace hf::runtime
