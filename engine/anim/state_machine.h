#pragma once
#include "anim/animation.h"
#include "anim/skeleton.h"
#include "math/math.h"

#include <string>
#include <vector>

// Slice BL — parameter-driven animation state machine + cross-fade blend, built PURELY on top of the
// existing skeletal-animation core (anim::SampleAnimation / anim::BlendAnimations). This is CPU-only:
// the FSM produces a joint-matrix palette exactly like SampleAnimation/BlendAnimations already do, so
// the unchanged GPU skinning path consumes its output with no new RHI / shader. ZERO backend symbols.
//
// Determinism: the FSM output (current state, transition progress, blend weight, final palette) is a
// PURE FUNCTION of (state graph, the scripted parameter timeline, the fixed dt sequence). There is no
// input / RNG / wall clock. The cross-fade weight is the deterministic ratio transitionTime/duration.
namespace hf::anim {

// One state in the machine: a named animation clip (by index into a caller-provided
// std::vector<Animation>) with a per-state playback `speed` (multiplies dt when accumulating the
// state's local time) and a `loop` flag (informational here — SampleAnimation already clamps to the
// clip range; kept for future looping/wrap policy and so the graph reads self-describingly).
struct AnimState {
    std::string name;
    int   animationIndex = -1;
    bool  loop = true;
    float speed = 1.0f;
};

// A parameter-threshold-gated edge between two states. `from == -1` means "any state" (an
// any-state transition fires from whichever state is current). `paramIndex` selects a float
// parameter; the edge is satisfied when (param `cmp` threshold) holds. `duration` is the cross-fade
// length in seconds (the blend weight ramps 0->1 over it).
struct Transition {
    enum class Cmp { Greater, Less };
    int   from = -1;
    int   to = -1;
    int   paramIndex = -1;
    Cmp   cmp = Cmp::Greater;
    float threshold = 0.0f;
    float duration = 0.0f;
};

// The state machine: owns states + transitions + a small named float parameter set, plus the runtime
// cursor {current, stateTime, transitioningTo (-1 = none), transitionTime}. Update(dt) advances the
// cursor deterministically; Evaluate() turns the cursor into a joint palette.
class StateMachine {
public:
    // --- Authoring -------------------------------------------------------------------------------
    // Add a state; returns its index (the order added == the index referenced by transitions).
    int AddState(const AnimState& state);
    // Add a transition (evaluated in the order added — first satisfied wins).
    void AddTransition(const Transition& t);
    // Declare a named float parameter (default 0); returns its index. Re-declaring a name returns the
    // existing index (and leaves its value unchanged).
    int AddParam(const std::string& name, float value = 0.0f);
    // Look up a parameter index by name (-1 if absent).
    int ParamIndex(const std::string& name) const;

    // Set the initial current state (resets stateTime + cancels any in-flight transition). Defaults
    // to 0 if never called and at least one state exists.
    void SetInitialState(int stateIndex);

    // --- Parameters (the deterministic drive) ----------------------------------------------------
    void  SetParam(const std::string& name, float value);  // no-op if the name is unknown
    void  SetParam(int paramIndex, float value);
    float GetParam(int paramIndex) const;

    // --- Tick ------------------------------------------------------------------------------------
    // Advance one fixed step. If NOT transitioning: stateTime += dt * speed(current), then the
    // outgoing transitions of `current` (and every from==-1 any-state edge) are evaluated IN THE
    // FIXED ORDER they were added; the FIRST whose condition holds begins a transition (records
    // transitioningTo, sets transitionTime=0, and seeds the target's time from 0). If transitioning:
    // BOTH the from-state's and to-state's local times advance, transitionTime += dt, and when
    // transitionTime >= duration the transition COMPLETES (current = target, stateTime carries the
    // target's accumulated time, transitioningTo = -1).
    void Update(float dt);

    // --- Output ----------------------------------------------------------------------------------
    // The joint-matrix palette for the current cursor. NOT transitioning -> SampleAnimation(current).
    // Transitioning -> BlendAnimations(from, to, weight = transitionTime/duration); weight 0 yields
    // the from-state pose, weight 1 the to-state pose (the existing BlendAnimations convention).
    std::vector<math::Mat4> Evaluate(const Skeleton& skeleton,
                                     const std::vector<Animation>& anims) const;

    // --- Accessors (tests / showcase) ------------------------------------------------------------
    int  Current() const { return current_; }
    int  TransitioningTo() const { return transitioningTo_; }
    bool IsTransitioning() const { return transitioningTo_ >= 0; }
    // Cross-fade weight in [0,1] (transitionTime/duration; 0 when not transitioning or duration<=0).
    float BlendWeight() const;
    const std::string& CurrentStateName() const;
    const std::string& TransitioningToName() const;  // returns "" when not transitioning

    int StateCount() const { return static_cast<int>(states_.size()); }

private:
    bool ConditionHolds(const Transition& t) const;

    struct Param { std::string name; float value = 0.0f; };

    std::vector<AnimState>  states_;
    std::vector<Transition> transitions_;
    std::vector<Param>      params_;

    int   current_ = 0;
    float stateTime_ = 0.0f;       // local time of the current (from) state
    int   transitioningTo_ = -1;   // -1 = not transitioning
    float transitionTime_ = 0.0f;  // elapsed cross-fade time
    float activeDuration_ = 0.0f;  // duration of the in-flight transition (the firing edge's)
    float toStateTime_ = 0.0f;     // local time of the target state during a transition
};

} // namespace hf::anim
