#include "llama-store-policy.h"

// store-policy skeleton (bench/THREE-TIER.md F13 S1): the shell owns one policy behind its
// interface and can hot-swap it for the M2 experiments without slab or io moving. Nobody else
// includes this unit; the shell controls both the creation and the swap.
uint8_t llama_store_policy_default::adopt_tier() const {
    return 1; // ram: the skeleton keeps adopted experts where the pre-store behavior had them
}
