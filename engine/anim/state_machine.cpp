#include "anim/state_machine.h"

#include <algorithm>

namespace hf::anim {

namespace {
const std::string kEmpty;
}  // namespace

int StateMachine::AddState(const AnimState& state) {
    states_.push_back(state);
    return static_cast<int>(states_.size()) - 1;
}

void StateMachine::AddTransition(const Transition& t) { transitions_.push_back(t); }

int StateMachine::AddParam(const std::string& name, float value) {
    for (size_t i = 0; i < params_.size(); ++i)
        if (params_[i].name == name) return static_cast<int>(i);
    params_.push_back({name, value});
    return static_cast<int>(params_.size()) - 1;
}

int StateMachine::ParamIndex(const std::string& name) const {
    for (size_t i = 0; i < params_.size(); ++i)
        if (params_[i].name == name) return static_cast<int>(i);
    return -1;
}

void StateMachine::SetInitialState(int stateIndex) {
    current_ = stateIndex;
    stateTime_ = 0.0f;
    transitioningTo_ = -1;
    transitionTime_ = 0.0f;
    toStateTime_ = 0.0f;
}

void StateMachine::SetParam(const std::string& name, float value) {
    int idx = ParamIndex(name);
    if (idx >= 0) params_[idx].value = value;
}

void StateMachine::SetParam(int paramIndex, float value) {
    if (paramIndex >= 0 && paramIndex < static_cast<int>(params_.size()))
        params_[paramIndex].value = value;
}

float StateMachine::GetParam(int paramIndex) const {
    if (paramIndex >= 0 && paramIndex < static_cast<int>(params_.size()))
        return params_[paramIndex].value;
    return 0.0f;
}

bool StateMachine::ConditionHolds(const Transition& t) const {
    if (t.paramIndex < 0 || t.paramIndex >= static_cast<int>(params_.size())) return false;
    const float v = params_[t.paramIndex].value;
    switch (t.cmp) {
        case Transition::Cmp::Greater: return v > t.threshold;
        case Transition::Cmp::Less:    return v < t.threshold;
    }
    return false;
}

void StateMachine::Update(float dt) {
    if (states_.empty()) return;

    if (transitioningTo_ < 0) {
        // --- Not transitioning: advance the current state, then look for an outgoing edge. -------
        const float speed = (current_ >= 0 && current_ < static_cast<int>(states_.size()))
                                ? states_[current_].speed : 1.0f;
        stateTime_ += dt * speed;

        // Evaluate transitions IN FIXED ADD-ORDER; the first satisfied one (matching `current_`, or
        // an any-state from==-1 edge) begins a transition. Deterministic: order is the add order.
        for (const Transition& t : transitions_) {
            if (t.from != -1 && t.from != current_) continue;
            if (t.to < 0 || t.to >= static_cast<int>(states_.size())) continue;
            if (t.to == current_) continue;  // a no-op self transition never "fires"
            if (ConditionHolds(t)) {
                transitioningTo_ = t.to;
                transitionTime_ = 0.0f;
                toStateTime_ = 0.0f;
                // Pin the FIRING edge's cross-fade duration on the cursor so BlendWeight/completion
                // use exactly this transition's duration (independent of edge-iteration later).
                activeDuration_ = t.duration;
                break;
            }
        }
    } else {
        // --- Transitioning: advance BOTH states' local times + the cross-fade clock. -------------
        const float fromSpeed = (current_ >= 0 && current_ < static_cast<int>(states_.size()))
                                    ? states_[current_].speed : 1.0f;
        const float toSpeed = (transitioningTo_ >= 0 &&
                               transitioningTo_ < static_cast<int>(states_.size()))
                                  ? states_[transitioningTo_].speed : 1.0f;
        stateTime_ += dt * fromSpeed;
        toStateTime_ += dt * toSpeed;
        transitionTime_ += dt;

        if (transitionTime_ >= activeDuration_) {
            // Complete: the target becomes current; carry the target's accumulated local time.
            current_ = transitioningTo_;
            stateTime_ = toStateTime_;
            transitioningTo_ = -1;
            transitionTime_ = 0.0f;
            toStateTime_ = 0.0f;
        }
    }
}

float StateMachine::BlendWeight() const {
    if (transitioningTo_ < 0 || activeDuration_ <= 0.0f) return 0.0f;
    return std::clamp(transitionTime_ / activeDuration_, 0.0f, 1.0f);
}

std::vector<math::Mat4> StateMachine::Evaluate(const Skeleton& skeleton,
                                               const std::vector<Animation>& anims) const {
    auto clipOf = [&](int stateIdx) -> const Animation* {
        if (stateIdx < 0 || stateIdx >= static_cast<int>(states_.size())) return nullptr;
        int ai = states_[stateIdx].animationIndex;
        if (ai < 0 || ai >= static_cast<int>(anims.size())) return nullptr;
        return &anims[ai];
    };

    if (transitioningTo_ < 0) {
        const Animation* clip = clipOf(current_);
        if (!clip) return std::vector<math::Mat4>(skeleton.joints.size(), math::Mat4::Identity());
        return SampleAnimation(skeleton, *clip, stateTime_);
    }

    const Animation* from = clipOf(current_);
    const Animation* to = clipOf(transitioningTo_);
    if (!from || !to)
        return std::vector<math::Mat4>(skeleton.joints.size(), math::Mat4::Identity());
    // weight 0 -> from, weight 1 -> to (the existing BlendAnimations convention).
    return BlendAnimations(skeleton, *from, stateTime_, *to, toStateTime_, BlendWeight());
}

const std::string& StateMachine::CurrentStateName() const {
    if (current_ >= 0 && current_ < static_cast<int>(states_.size())) return states_[current_].name;
    return kEmpty;
}

const std::string& StateMachine::TransitioningToName() const {
    if (transitioningTo_ >= 0 && transitioningTo_ < static_cast<int>(states_.size()))
        return states_[transitioningTo_].name;
    return kEmpty;
}

} // namespace hf::anim
