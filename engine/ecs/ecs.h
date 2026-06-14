#pragma once
// Hazard Forge — entity-component-system runtime (data-oriented core).
//
// A pragmatic, REAL ECS: a registry of entities, one dense component POOL per type, and
// multi-component VIEWS that iterate the entities holding all of a given set of components.
// Pool-based (sparse-set) rather than archetype-chunked — simpler, still cache-friendly for
// dense iteration, and a natural base to evolve toward archetypes later.
//
// Backend-agnostic: depends ONLY on the C++ standard library. There are NO vk*/Metal/rhi
// symbols in this header — components may hold rhi/backend pointers as opaque values, but ecs.h
// itself only ever sees std types and the component types the caller passes through its templates.
//
//   Entity        : index + generation handle; stale handles are detectable via generation.
//   Registry      : create()/destroy()/valid(); add/get/has/remove<T>; view<A,B,...>().
//   ComponentPool : dense vector of components + a sparse entity->dense index map (cache-friendly).
//   View<A,...>   : range-for iterable yielding (Entity, A&, B&, ...) for entities holding all of them.
//
// No RTTI and no exceptions on the hot path (add/get/has/remove/view iteration are all noexcept-able
// straight-line code; only out-of-the-ordinary misuse asserts in debug builds).
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace hf::ecs {

// ---------------------------------------------------------------------------
// Entity handle: a 32-bit index + 32-bit generation packed into 64 bits. The generation is
// bumped every time an index slot is recycled, so a handle to a destroyed entity (same index,
// older generation) is detected as stale by Registry::valid().
// ---------------------------------------------------------------------------
struct Entity {
    uint32_t index = 0;
    uint32_t generation = 0;

    bool operator==(const Entity& o) const { return index == o.index && generation == o.generation; }
    bool operator!=(const Entity& o) const { return !(*this == o); }
};

// A reserved index marking "no entity".
inline constexpr uint32_t kInvalidIndex = ~0u;
inline constexpr Entity kNullEntity{kInvalidIndex, 0};

// ---------------------------------------------------------------------------
// Component storage: a sparse-set pool. Components live in a DENSE vector (cache-friendly to
// iterate). A sparse vector maps entity.index -> dense slot; a parallel dense vector maps each
// dense slot back to its owning entity (so iteration yields the entity, and removal can swap-pop).
// ---------------------------------------------------------------------------

// Type-erased base so the registry can hold a heterogeneous set of pools and destroy an entity's
// components across all of them without knowing each concrete type.
struct IComponentPool {
    virtual ~IComponentPool() = default;
    // Remove the component for entity index `idx` if present (used by Registry::destroy()).
    virtual void RemoveByIndex(uint32_t idx) = 0;
};

template <typename T>
class ComponentPool final : public IComponentPool {
public:
    bool Has(uint32_t idx) const {
        return idx < sparse_.size() && sparse_[idx] != kInvalidIndex;
    }

    // Add (or overwrite) the component for entity index `idx`. Returns a reference to the stored value.
    T& Add(uint32_t idx, T value) {
        if (idx >= sparse_.size()) sparse_.resize(idx + 1, kInvalidIndex);
        if (sparse_[idx] != kInvalidIndex) {
            // Overwrite in place — keeps the dense slot stable.
            dense_[sparse_[idx]] = std::move(value);
            return dense_[sparse_[idx]];
        }
        uint32_t dense = static_cast<uint32_t>(dense_.size());
        sparse_[idx] = dense;
        dense_.push_back(std::move(value));
        owners_.push_back(idx);
        return dense_.back();
    }

    T& Get(uint32_t idx) {
        assert(Has(idx) && "ComponentPool::Get on an entity without this component");
        return dense_[sparse_[idx]];
    }
    const T& Get(uint32_t idx) const {
        assert(Has(idx) && "ComponentPool::Get on an entity without this component");
        return dense_[sparse_[idx]];
    }

    void RemoveByIndex(uint32_t idx) override {
        if (!Has(idx)) return;
        uint32_t dense = sparse_[idx];
        uint32_t last = static_cast<uint32_t>(dense_.size()) - 1;
        if (dense != last) {
            // Swap-and-pop: move the last element into the freed slot, fix up its owner mapping.
            dense_[dense] = std::move(dense_[last]);
            owners_[dense] = owners_[last];
            sparse_[owners_[dense]] = dense;
        }
        dense_.pop_back();
        owners_.pop_back();
        sparse_[idx] = kInvalidIndex;
    }

    // Dense iteration surface (used by View).
    std::size_t Size() const { return dense_.size(); }
    const std::vector<uint32_t>& Owners() const { return owners_; }
    T& DenseAt(std::size_t d) { return dense_[d]; }

private:
    std::vector<uint32_t> sparse_;  // entity.index -> dense slot (or kInvalidIndex)
    std::vector<uint32_t> owners_;  // dense slot   -> entity.index
    std::vector<T> dense_;          // the components themselves, packed
};

// ---------------------------------------------------------------------------
// Registry: owns entities and the per-type component pools.
// ---------------------------------------------------------------------------
class Registry {
public:
    // --- Entity lifecycle ----------------------------------------------------

    Entity create() {
        uint32_t idx;
        if (!freeList_.empty()) {
            idx = freeList_.back();
            freeList_.pop_back();
            alive_[idx] = true;
        } else {
            idx = static_cast<uint32_t>(generations_.size());
            generations_.push_back(0);
            alive_.push_back(true);
        }
        return Entity{idx, generations_[idx]};
    }

    bool valid(Entity e) const {
        return e.index < generations_.size() && alive_[e.index] &&
               generations_[e.index] == e.generation;
    }

    void destroy(Entity e) {
        if (!valid(e)) return;
        // Drop every component this entity owns, across all pools.
        for (auto& slot : pools_)
            if (slot) slot->RemoveByIndex(e.index);
        alive_[e.index] = false;
        ++generations_[e.index];  // invalidate any outstanding handle to this slot
        freeList_.push_back(e.index);
    }

    std::size_t aliveCount() const {
        std::size_t n = 0;
        for (bool a : alive_) if (a) ++n;
        return n;
    }

    // --- Components ----------------------------------------------------------

    template <typename T>
    T& add(Entity e, T value) {
        assert(valid(e) && "Registry::add on an invalid entity");
        return pool<T>().Add(e.index, std::move(value));
    }

    template <typename T>
    bool has(Entity e) const {
        if (!valid(e)) return false;
        const ComponentPool<T>* p = poolPtr<T>();
        return p && p->Has(e.index);
    }

    template <typename T>
    T& get(Entity e) {
        assert(valid(e) && "Registry::get on an invalid entity");
        return pool<T>().Get(e.index);
    }

    template <typename T>
    const T& get(Entity e) const {
        assert(valid(e) && "Registry::get on an invalid entity");
        return poolPtr<T>()->Get(e.index);
    }

    template <typename T>
    void remove(Entity e) {
        if (!valid(e)) return;
        if (ComponentPool<T>* p = poolPtr<T>()) p->RemoveByIndex(e.index);
    }

    // --- Views ---------------------------------------------------------------
    // view<A, B, ...>() iterates every entity that has ALL of A, B, .... It walks the SMALLEST
    // pool among the requested types and checks the rest, so iteration cost scales with the
    // rarest component. Each step yields a tuple (Entity, A&, B&, ...).
    template <typename... Ts>
    class View {
        static_assert(sizeof...(Ts) >= 1, "view<> needs at least one component type");

    public:
        explicit View(Registry& reg) : reg_(&reg) {}

        class Iterator {
        public:
            Iterator(Registry* reg, const std::vector<uint32_t>* driver, std::size_t pos,
                     std::size_t end)
                : reg_(reg), driverOwners_(driver), pos_(pos), end_(end) {
                Advance();
            }

            bool operator!=(const Iterator& o) const { return pos_ != o.pos_; }
            void operator++() { ++pos_; Advance(); }

            std::tuple<Entity, Ts&...> operator*() const {
                Entity e = reg_->entityAt((*driverOwners_)[pos_]);
                return std::tuple<Entity, Ts&...>(e, reg_->template get<Ts>(e)...);
            }

        private:
            void Advance() {
                // Skip dense slots in the driver pool whose entity lacks one of the OTHER types.
                while (pos_ != end_) {
                    Entity e = reg_->entityAt((*driverOwners_)[pos_]);
                    if (reg_->valid(e) && (reg_->template has<Ts>(e) && ...)) break;
                    ++pos_;
                }
            }

            Registry* reg_;
            const std::vector<uint32_t>* driverOwners_;
            std::size_t pos_;
            std::size_t end_;
        };

        Iterator begin() {
            Resolve();
            return Iterator(reg_, driverOwners_, 0, driverSize_);
        }
        Iterator end() {
            Resolve();
            return Iterator(reg_, driverOwners_, driverSize_, driverSize_);
        }

    private:
        // Pick the smallest requested pool as the iteration driver.
        void Resolve() {
            if (resolved_) return;
            resolved_ = true;
            const std::vector<uint32_t>* owners[] = {ownersOf<Ts>()...};
            std::size_t sizes[] = {sizeOf<Ts>()...};
            std::size_t best = 0;
            for (std::size_t i = 1; i < sizeof...(Ts); ++i)
                if (sizes[i] < sizes[best]) best = i;
            driverOwners_ = owners[best];
            driverSize_ = sizes[best];
        }

        template <typename U>
        const std::vector<uint32_t>* ownersOf() const {
            static const std::vector<uint32_t> kEmpty;
            const ComponentPool<U>* p = reg_->template poolPtr<U>();
            return p ? &p->Owners() : &kEmpty;
        }
        template <typename U>
        std::size_t sizeOf() const {
            const ComponentPool<U>* p = reg_->template poolPtr<U>();
            return p ? p->Size() : 0;
        }

        Registry* reg_;
        bool resolved_ = false;
        const std::vector<uint32_t>* driverOwners_ = nullptr;
        std::size_t driverSize_ = 0;
    };

    template <typename... Ts>
    View<Ts...> view() { return View<Ts...>(*this); }

    // Reconstruct the live Entity handle for an entity index (used by views).
    Entity entityAt(uint32_t idx) const { return Entity{idx, generations_[idx]}; }

private:
    // Stable per-type id, assigned lazily on first use of each component type.
    template <typename T>
    static std::size_t TypeId() {
        static const std::size_t id = NextTypeId();
        return id;
    }
    static std::size_t NextTypeId() {
        static std::size_t counter = 0;
        return counter++;
    }

    template <typename T>
    ComponentPool<T>& pool() {
        std::size_t id = TypeId<T>();
        if (id >= pools_.size()) pools_.resize(id + 1);
        if (!pools_[id]) pools_[id] = std::make_unique<ComponentPool<T>>();
        return *static_cast<ComponentPool<T>*>(pools_[id].get());
    }

    template <typename T>
    ComponentPool<T>* poolPtr() {
        std::size_t id = TypeId<T>();
        if (id >= pools_.size() || !pools_[id]) return nullptr;
        return static_cast<ComponentPool<T>*>(pools_[id].get());
    }
    template <typename T>
    const ComponentPool<T>* poolPtr() const {
        std::size_t id = TypeId<T>();
        if (id >= pools_.size() || !pools_[id]) return nullptr;
        return static_cast<const ComponentPool<T>*>(pools_[id].get());
    }

    std::vector<uint32_t> generations_;  // per index: current generation
    std::vector<bool> alive_;            // per index: is the slot currently a live entity
    std::vector<uint32_t> freeList_;     // recycled indices
    std::vector<std::unique_ptr<IComponentPool>> pools_;  // per component type id
};

}  // namespace hf::ecs
