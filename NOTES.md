# Notes


At a high level, this engine is trying to serve multiple autoregressive
generation requests efficiently.

A single request looks like:

1. Read a prompt.
2. Run the model on the prompt.
3. Produce the first generated token.
4. Feed that token back into the model.
5. Produce the next token.
6. Repeat until EOS or `max_new_tokens`.

The naive implementation is:

- give each request its own dedicated KV cache
- run each request independently
- let prompt processing be a one-off forward pass
- let decode be a separate loop per request

That is easy to understand but inefficient.

This engine instead tries to share work in three ways:

- **paged KV storage**: all requests draw from one common pool of KV blocks
- **continuous batching**: active requests are grouped and advanced together
- **prefix reuse**: full prompt blocks can be reused across requests if their
  content matches

The design is intentionally split into host-side logic and device-side logic.

- Host-side logic decides *which* requests to run, *which* blocks belong to
  which request, and *what* the current runtime metadata is.
- Device-side logic does the expensive tensor work: attention, cache writes,
  cache gathers, and logits.

This separation is one of the most important ideas in the codebase.

## The Most Important Concept: Two Kinds of State

The engine has two very different kinds of state.

### 1. Host runtime state

This is light-weight metadata:

- current positions
- last generated tokens
- block table rows
- which request is in which slot
- whether a slot is prefilling or decoding
- which physical blocks a slot owns
- prefix-cache bookkeeping

This data is small, branchy, and control-flow heavy. It changes often and is
best managed in ordinary Python/Numpy code.

### 2. Device model/cache state

This is the expensive state:

- the paged KV tensors
- layer-by-layer K/V storage for all active and reusable blocks

This is large, array-oriented, and suitable for JAX/XLA compilation.

One way to understand the refactor is:

- `runtime.py` owns host runtime state
- `state.py` owns JAX-facing cache state

That split is deliberate and foundational.

## The Core Serving Strategy

The engine has two operational phases for each request.

### Prefill

Prefill means: process prompt tokens and write their K/V into cache.

In this engine, prefill is **chunked** by `block_size`.

Instead of one giant prompt pass, we process one fixed-size block at a time.

Why?

- stable shapes are easier to JIT well
- prompt length no longer changes the kernel shape
- paging and prefix caching line up naturally with fixed-size prompt blocks

### Decode

Decode means: given the last generated token and the cache, produce one new
token.

Decode is the steady-state serving loop. Once a request enters decode, it
advances one token at a time.

This engine packs active decode slots into a batch bucket such as `1, 2, 4, 8,
...`, so the GPU step depends on the number of active requests rather than
always paying for `max_num_seqs`.

## Slot Mental Model

A **slot** is the engine’s unit of scheduling.

You can think of a slot as:

- one lane in the continuous-batching engine
- one row in the block table
- one current request, if occupied

Slots are not requests themselves. Requests move through slots.

A slot can be:

- free
- prefilling
- decoding

Each live slot has:

- a request
- a list of physical block ids
- a list of hashes for the full blocks already registered in the prefix cache
- the next prompt position to prefill
- the number of logical blocks already filled

That bundle lives in `SlotState`.

## File-by-File Overview

## `jllm/engine/request.py`

This file defines the small data containers the rest of the engine passes
around.

### `SamplingParams`

This is the minimal generation policy currently supported by the engine.

It contains:

- `max_new_tokens`
- optional `eos_id`

There is no sampling stack yet. This is greedy generation.

### `Request`

This is the user-facing unit of work once admitted into the engine.

It contains:

- `request_id`
- prompt token ids
- sampling params
- accumulated output token ids
- finished flag
- current slot id, if any

This object is mutable because it is a natural “current request record.”

### `StepEvent`

This is what the engine yields to stream consumers.

Each event says:

- which request produced output
- which token was produced
- whether that request is now finished

### `EngineStats`

This is observability for the engine loop.

It tracks:

- number of admissions
- scheduler loop count
- number of prefill batches
- number of decode batches
- number of extend calls
- number of decode calls
- emitted tokens
- cache-hit blocks

This is useful for tests and later for benchmark interpretation.

## `jllm/engine/runtime.py`

This file defines the **host runtime state**.

It is intentionally functional in style:

- plain dataclasses
- top-level helper functions
- copy-on-write updates

This matches the rest of the repo better than class-style methods.

### `SlotState`

This is per-live-slot Python metadata.

Important fields:

- `slot`: the integer slot index
- `request`: the request occupying that slot
- `phase`: `"prefilling"` or `"decoding"`
- `blocks`: physical block ids currently owned or referenced by the slot
- `hashes`: content hashes for full logical blocks already known
- `next_prefill_pos`: next prompt index to prefill
- `filled_blocks`: how many logical blocks have been wired in

This structure is not JAX-traced. It is scheduler metadata.

### `RuntimeState`

This is the compact array-based host state.

It contains:

- `positions`
- `last_tokens`
- `block_tables`

These are the three arrays the engine most often needs to construct packed
kernel inputs.

### Update helpers

The helper functions are:

- `init_runtime_state`
- `with_cached_prefix`
- `with_block`
- `with_decode_state`
- `without_slot`

These functions return new `RuntimeState` objects rather than mutating the old
one.

Why is that reasonable?

- these arrays are small compared with the KV cache
- it makes reasoning simpler
- tests can compare before/after states more easily
- the engine’s control flow becomes less stateful in surprising ways

## `jllm/engine/cache.py`

This file is a host-side helper layer for prefix caching.

It exists because prefix caching is logically separate from the scheduler.

The scheduler wants to ask:

- does this prompt share a cached prefix?
- if yes, which blocks?
- when a block becomes full, how do I register it?
- when a request finishes, how do I release its blocks?

`cache.py` answers those questions without forcing `engine.py` to know the
details of hash construction.

### `CachedPrefix`

This is a simple record:

- `block_ids`
- `hashes`

It represents the reusable full-block prefix already present in the cache.

### `prompt_block_hashes`

This computes the full-block Merkle-chain hashes for a prompt.

Important detail:

- only full blocks are hashed
- trailing partial blocks are ignored

That means the cache key space is block-aligned.

### `take_cached_prefix`

This is the main “admit-time lookup” function.

It:

1. computes full-block hashes for the prompt
2. walks the hashes from the beginning
3. stops at the first missing block
4. `touch`es every reused block

The prefix must be contiguous from the beginning. This is important. The engine
does not support arbitrary interior block reuse.

### `register_full_block`

When the engine fills a complete logical block, this helper computes the block’s
Merkle-chain hash and registers it in the global cache.

### `release_blocks`

This is a thin wrapper around block-manager release semantics.

## `jllm/engine/paged.py`

This file contains two different things:

1. the **device layout** of the paged KV cache
2. the **host-side block manager**

That is why it is one of the most conceptually dense files.

## Device-side part

### `PagedLayerCache`

One transformer layer’s K/V storage:

- `k`
- `v`

Shape:

- `[num_blocks, block_size, num_kv_heads, head_dim]`

### `PagedCache`

The full model cache:

- list of `PagedLayerCache`
- shared `block_size`

### `init_paged_cache`

Allocates the full K/V pool for all layers.

Conceptually:

- every layer gets the same block layout
- different requests are separated by block tables, not by separate cache
  tensors

### Scatter/gather primitives

The three essential cache operations are:

- `scatter_kv_decode`
- `scatter_kv_prefill_batch`
- `gather_kv`

#### `scatter_kv_decode`

Writes one token per active decode row to its `(phys_block, slot_in_block)`.

#### `scatter_kv_prefill_batch`

Writes one full `block_size` chunk per prefill row.

This is the bridge between chunked prefill and the paged pool.

#### `gather_kv`

Turns block-table rows back into a flat `[B, NB*block_size, H, D]` view.

This is how attention sees the logical prefix even though storage is physically
paged.

## Host-side block manager

### Why a block manager exists

The GPU cache is just a tensor pool. It does not know:

- which blocks are free
- which ones are reusable
- which ones should be evicted

That is the block manager’s job.

### `KVCacheBlock`

Per-physical-block metadata:

- block id
- optional content hash
- refcount
- doubly-linked free-list pointers

### `BlockManager`

Global cache metadata:

- pool of blocks
- head/tail of free queue
- `hash_to_block` lookup

### Important policy choices

#### Sentinel block 0

Block 0 is special:

- never freed
- never cached
- safe target for padded dummy writes

This simplifies bucket padding because inactive rows can still produce legal
indices.

#### Refcounting

A block is only on the free list when `ref_cnt == 0`.

If a cached block is reused:

- `touch` increments refcount
- if it was currently free, it is removed from the free list

#### LRU eviction

`alloc_new` pops from the free-list head.

That means the least-recently-freed block is reused first.

#### Reverse-order release

When releasing a chain of blocks, the code appends them to the free queue in
reverse order.

This means:

- leaf blocks become evictable first
- shared prefix blocks become evictable later

That matches the intuition that common prefixes are more valuable to keep.

## `jllm/engine/state.py`

This file is now intentionally small.

Its purpose is:

- define the JAX-facing `EngineState`
- provide thin wrappers around the JIT kernels

### `EngineState`

This now contains only:

- `cache`

That is a deliberate simplification. Previously, more runtime metadata lived in
the JAX state, but that mixed control state with expensive array state.

### `prefill`

This function takes:

- the generic decoder model
- current cache state
- packed chunk ids
- packed positions
- packed block tables
- packed physical blocks

It returns:

- updated cache state
- logits for each row and each token position in the prefill chunk

### `decode`

This function takes the packed active decode batch and returns:

- updated cache state
- one new token per active row

This file is intentionally a boundary adapter, not a logic-heavy module.

## `jllm/engine/generate.py`

This file contains the actual kernel definitions.

It is easy to misread this file as “the model,” but that is not quite right.

It is better to think of it as:

- **model-generic paged execution**

The engine needs special logic here because the model is being run against
paged KV storage rather than against a simple dense sequence tensor.

### Why this belongs in engine, not model

The model layer knows:

- how attention works
- how layers are applied
- what the projections and norms are

The engine layer knows:

- how K/V are physically stored
- how to scatter into blocks
- how to gather from block tables
- how to batch active slots

So `generate.py` is really the point where those two worlds meet.

### Why this file should not know about Qwen specifically

That was one of the motivations for the recent abstraction cleanup.

Now `generate.py` depends on:

- `DecoderOnlyModel`
- `Attention`
- `DecoderLayer`

from `jllm.model.common`, rather than on Qwen classes directly.

That is the right direction because the engine does not fundamentally care about
Qwen; it cares about a decoder-only model with the required structural fields.

### `attention_prefill`

This function:

1. projects Q/K/V for the current prefill chunk
2. applies optional Q/K norm
3. applies RoPE
4. scatters the new K/V into the paged cache
5. gathers the full logical prefix from block tables
6. runs masked attention
7. returns the attention output plus updated layer cache

This is the heart of chunked prefill.

### `decoder_layer_prefill`

This is a standard transformer layer written against paged prefill attention:

- input norm
- attention
- residual
- post-attention norm
- MLP
- residual

### `prefill_step`

This is the batched prefill kernel.

Each row corresponds to:

- one active prefilling slot
- one `block_size` prompt chunk
- one fresh physical block

Why return logits for all positions in the chunk?

Because on the final prefill chunk, the engine needs the logits for the last
real prompt token in order to produce the first generated token.

### `attention_decode_cb`

This is the decode analogue of `attention_prefill`.

Differences:

- each row has only one query token
- cache write is one token at one slot position
- the mask uses the row’s current decode position

### `decode_step_cb`

This is the steady-state continuous-batching decode kernel.

One row in, one token out, all active decode rows together.

### JIT wrappers

The module ends with:

- `prefill_step_jit`
- `decode_step_cb_jit`

Both use donation because the cache is large and the kernel is naturally a
state transition on that cache.

## `jllm/engine/engine.py`

This is the scheduler and orchestration layer.

If you want to understand system behavior, this is the most important file.

## What `Driver` is

`Driver` is the live engine instance.

It holds:

- model
- configuration limits
- JAX cache state
- host runtime state
- block manager
- live requests
- waiting queue
- streaming queues
- scheduler stats
- background thread state

It is mutable because it is the running service object.

This is an intentional compromise:

- inner runtime state is functional
- outer orchestration shell is mutable

That is a pragmatic and understandable split.

## Request lifecycle

The lifecycle of a request is:

1. `add_request`
2. placed in `waiting`
3. admitted into a free slot
4. prefills in chunks, unless fully cached
5. transitions to decode
6. emits one token per decode step
7. finishes on EOS or token limit
8. slot is released and blocks are decremented

## `make_driver`

This function enforces engine invariants:

- `max_prefill_len <= max_model_len`
- lengths must be block-aligned
- `num_blocks` must be large enough for the worst case

The important design choice here is that undersized cache is rejected up front.
There is no preemption policy yet.

## `step`

This is the single most important scheduler function.

It does:

1. `_admit`
2. if decode slots exist:
   - `_decode`
   - `_prefill_batch`
3. else:
   - `_prefill_batch`

This means decode is prioritized when decode work exists.

Why?

- decode drives throughput
- decode requests are latency-sensitive once generation has started
- prefill still makes progress, but decode is not stalled behind prompt work

## `_admit`

This function moves waiting requests into free slots.

For each admitted request it:

1. looks up the cached full-block prefix
2. creates a `SlotState`
3. wires cached blocks into the runtime block table row
4. updates stats
5. if the prompt is fully cached, jumps directly to decode

That last case is important.

If the prompt is fully cached, no prefill kernel is needed. The request enters
decode immediately.

## `_prefill_batch`

This function builds the packed prefill batch.

Key steps:

1. select prefilling slots
2. choose a batch bucket
3. allocate one new physical block per active slot
4. populate packed arrays
5. call `state.prefill(...)`
6. register any newly completed full blocks
7. if a slot finished its prompt:
   - pick the first generated token from the last real prompt position
   - switch the slot to decode
   - emit a `StepEvent`

The important idea is that one scheduler step can advance multiple prefilling
slots at once.

## `_decode`

This function builds the packed decode batch.

Key steps:

1. select decoding slots
2. choose a batch bucket
3. allocate new block on logical block boundary crossings
4. populate packed arrays
5. call `state.decode(...)`
6. update runtime positions and last tokens
7. register hashes for newly completed full decode blocks
8. emit `StepEvent`s
9. release completed requests

This is the steady-state throughput path.

## `_release`

This tears down a slot:

- remove it from `running`
- mark request finished
- release block references
- clear runtime row

This is where host state and block-manager state are brought back into
consistency after a request ends.

## A Full End-to-End Walkthrough

Suppose a new request arrives with a 40-token prompt, `block_size=16`, and
`max_new_tokens=8`.

### Admit phase

The prompt has:

- 2 full blocks
- 1 partial block of 8 tokens

The engine:

1. hashes the first two full blocks
2. checks whether those hashes are already cached
3. admits the request to a slot

If the first block is cached and the second is not:

- the slot starts with one inherited block
- `next_prefill_pos = 16`
- `filled_blocks = 1`

### Prefill phase

On the first prefill batch for this slot:

- the engine allocates a fresh block
- writes prompt tokens `[16:32]`
- registers the block hash because the block is full

On the next prefill batch:

- the engine allocates another fresh block
- writes prompt tokens `[32:40]` plus harmless padding
- does **not** register a hash, because the block is partial

Because the prompt is now exhausted:

- the engine reads logits at the last real prompt token
- picks the first generated token
- transitions the slot to decode

### Decode phase

Now every decode step:

- uses `last_tokens[slot]` as input
- writes one new K/V position
- gathers the full prefix through the block table
- produces one new token

If decode fills the remainder of the last partial block so that it becomes
full, the engine registers a hash for that block at that time.

### Finish phase

When EOS or token limit is hit:

- blocks are released
- the slot becomes free
- reusable full blocks remain in the cache if their refcount reaches zero

## Why the Scheduler is a Little Complicated

Inference engines are fundamentally awkward because they combine:

- dynamic lifetimes
- irregular lengths
- expensive compiled kernels
- memory reuse
- fairness concerns

That is why the code is split instead of written as one monolithic loop.

The split aims to keep each complexity local:

- `runtime.py`: host runtime representation
- `cache.py`: prefix-cache logic
- `paged.py`: low-level storage and reuse policy
- `state.py`: JAX boundary
- `generate.py`: paged execution kernels
- `engine.py`: orchestration

## What to Read First

If you are new to the code, this order is a good learning path:

1. `request.py`
2. `runtime.py`
3. `cache.py`
4. `paged.py`
5. `state.py`
6. `engine.py`
7. `generate.py`

That order starts from simple containers and ends with the densest tensor code.

## Current Limitations

A few important things are intentionally not implemented yet.

- no preemption if the cache is undersized
- no sampling stack beyond greedy argmax
- no speculative decoding
- no custom fused kernels
- no multi-node serving

That is useful context when reading the code. Many design decisions make more
sense once you remember the project is optimizing for clarity and experimental
iteration first, rather than completeness.

## Final Mental Model

If you want one short summary to keep in your head, use this:

- `engine.py` decides **who runs**
- `runtime.py` records **where each request is**
- `cache.py` decides **what prefix can be reused**
- `paged.py` defines **how KV memory is stored**
- `state.py` is the **boundary between Python scheduling and JAX kernels**
- `generate.py` executes **decoder layers against paged KV storage**

That is the architecture in one sentence.
