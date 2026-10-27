#pragma once

#include <cstdint>
#include <memory>

// store-policy (bench/THREE-TIER.md F13): the only part that moves with the A/B measurements.
// Hot-swappable through this interface while slab and io do not move. The M2a policy does not move
// anything: adoptions land on RAM, and the residency decisions arrive when a transfer executor exists.
class llama_store_policy {
public:
    virtual ~llama_store_policy() = default;

    // tire tiers: 0 = vram, 1 = ram, 2 = ssd (mirror of the store shell tier naming)
    virtual uint8_t adopt_tier() const = 0;
};

// default policy of the skeleton: keep everything where the pre-store behavior had it, host RAM
class llama_store_policy_default final : public llama_store_policy {
public:
    uint8_t adopt_tier() const override;
};
