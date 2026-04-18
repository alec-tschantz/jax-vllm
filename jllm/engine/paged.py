"""Paged KV cache primitives + BlockManager with content-hash prefix caching.

Device-side primitives (unchanged layout): one pool of fixed-size blocks; sequences
own block tables of physical block indices.

- scatter_kv_decode: write 1 token of K/V per batch element to (phys_block, slot)
- scatter_kv_prefill: write T tokens of a single prompt into ceil(T/block_size) blocks
- gather_kv: reassemble `[B, NB*block_size, H, D]` views from block_tables

Host-side BlockManager: vLLM-style content-addressed cache.

- KVCacheBlock per physical block, with `block_hash`, `ref_cnt`, and doubly-
  linked free-queue pointers.
- Block hashes form a Merkle chain: `hash(parent_hash_bytes || token_id_bytes)`
  (sha256). Only full blocks (exactly block_size tokens) are ever hashed.
- `hash_to_block: dict[BlockHash, int]` is the global lookup for cached blocks.
- Reference counting: a block is only on the free queue when `ref_cnt == 0`.
  Reused cached blocks are "touched" (ref_cnt++, unlinked from free queue).
- Eviction is LRU: `alloc_new` pops from the head (least-recently-freed); if the
  popped block has a hash, the hash map entry is removed.
- `release` appends freed blocks to the tail in reverse order (leaf-last = root
  of the prefix ends up deepest in the queue = evicted last). This matches
  vLLM's "evict chain tail first" rule.
- Block 0 is reserved as an idle-slot sentinel: `ref_cnt=inf`, never on the free
  queue, never cached. Other slots' scatters to (0,0) are harmless garbage.
"""
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Iterable, Optional

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..model.common import KVCacheConfig

BlockHash = bytes
SENTINEL_REF = 1 << 30  # "infinite" ref count for the sentinel block


# ---------- device-side pool ----------


class PagedLayerCache(eqx.Module):
    k: Array  # [num_blocks, block_size, num_kv_heads, head_dim]
    v: Array


class PagedCache(eqx.Module):
    layers: list[PagedLayerCache]
    block_size: int


def init_paged_cache(
    cfg: KVCacheConfig, num_blocks: int, block_size: int, dtype
) -> PagedCache:
    shape = (num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim)
    return PagedCache(
        layers=[
            PagedLayerCache(k=jnp.zeros(shape, dtype=dtype), v=jnp.zeros(shape, dtype=dtype))
            for _ in range(cfg.num_hidden_layers)
        ],
        block_size=block_size,
    )


# ---------- content hashing ----------


def compute_block_hash(parent_hash: BlockHash, token_ids: Iterable[int]) -> BlockHash:
    """Merkle-chain hash. Parent is `b""` for the first block of a sequence.
    Deterministic across runs: little-endian uint32 serialisation, no pickle."""
    h = hashlib.sha256()
    h.update(parent_hash)
    for tok in token_ids:
        h.update(struct.pack("<I", int(tok)))
    return h.digest()


# ---------- block manager ----------


@dataclass
class KVCacheBlock:
    block_id: int
    block_hash: Optional[BlockHash] = None
    ref_cnt: int = 0
    prev_free: Optional["KVCacheBlock"] = None
    next_free: Optional["KVCacheBlock"] = None


@dataclass
class BlockManager:
    pool: list[KVCacheBlock]
    free_head: Optional[KVCacheBlock] = None
    free_tail: Optional[KVCacheBlock] = None
    hash_to_block: dict[BlockHash, int] = field(default_factory=dict)


def make_manager(num_blocks: int) -> BlockManager:
    """Pre-allocate the pool, reserve block 0 as the idle-slot sentinel, and
    thread blocks 1..num_blocks-1 into the free queue (head=1, tail=last)."""
    if num_blocks < 2:
        raise ValueError(f"num_blocks must be >= 2 (1 sentinel + >=1 usable)")
    pool = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
    pool[0].ref_cnt = SENTINEL_REF  # never enters free queue

    # Link blocks 1..num_blocks-1 as a doubly-linked free list.
    for i in range(1, num_blocks - 1):
        pool[i].next_free = pool[i + 1]
        pool[i + 1].prev_free = pool[i]
    mgr = BlockManager(
        pool=pool,
        free_head=pool[1],
        free_tail=pool[num_blocks - 1],
    )
    return mgr


def _unlink(mgr: BlockManager, block: KVCacheBlock) -> None:
    p, n = block.prev_free, block.next_free
    if p is not None:
        p.next_free = n
    else:
        mgr.free_head = n
    if n is not None:
        n.prev_free = p
    else:
        mgr.free_tail = p
    block.prev_free = None
    block.next_free = None


def _append_tail(mgr: BlockManager, block: KVCacheBlock) -> None:
    block.prev_free = mgr.free_tail
    block.next_free = None
    if mgr.free_tail is not None:
        mgr.free_tail.next_free = block
    else:
        mgr.free_head = block
    mgr.free_tail = block


def touch(mgr: BlockManager, block_id: int) -> None:
    """Reuse a cached block: inc ref_cnt; unlink from free queue if ref was 0."""
    block = mgr.pool[block_id]
    if block.ref_cnt == 0:
        _unlink(mgr, block)
    block.ref_cnt += 1


def alloc_new(mgr: BlockManager) -> int:
    """Pop LRU from free queue (head); evict cached hash if present; return id
    with ref_cnt = 1."""
    block = mgr.free_head
    if block is None:
        raise MemoryError("no free blocks available")
    _unlink(mgr, block)
    if block.block_hash is not None:
        mgr.hash_to_block.pop(block.block_hash, None)
        block.block_hash = None
    block.ref_cnt = 1
    return block.block_id


def release(mgr: BlockManager, block_ids: Iterable[int]) -> None:
    """Decrement ref counts; append to free tail in reverse order so leaf (last)
    blocks evict first and prefix (first) blocks evict last — matches vLLM's
    'chain tail first' rule.

    The sentinel block (ref_cnt == SENTINEL_REF) is skipped."""
    ids = list(block_ids)
    for block_id in reversed(ids):
        block = mgr.pool[block_id]
        if block.ref_cnt >= SENTINEL_REF:
            continue
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            _append_tail(mgr, block)


def register_hash(mgr: BlockManager, block_id: int, block_hash: BlockHash) -> None:
    """Register a now-full block's hash. setdefault semantics: if a different
    block already maps to this hash, leave the existing mapping alone (v1's
    'tolerate duplicates' rule — avoids v0's block-table rewrite)."""
    block = mgr.pool[block_id]
    if block.block_hash is not None:
        return  # already registered
    if block_hash in mgr.hash_to_block:
        return  # a different block already holds this hash; accept duplication
    mgr.hash_to_block[block_hash] = block_id
    block.block_hash = block_hash


def lookup(mgr: BlockManager, block_hash: BlockHash) -> Optional[int]:
    """Return block_id for a cached hash, or None if not cached."""
    return mgr.hash_to_block.get(block_hash)


# ---------- device-side primitives (unchanged) ----------


def scatter_kv_decode(
    layer: PagedLayerCache,
    k_new: Array,           # [B, 1, H_kv, D]
    v_new: Array,
    phys_block: Array,      # [B] int32
    slot_in_block: Array,   # [B] int32
) -> PagedLayerCache:
    """Write one token per batch element to (phys_block[b], slot_in_block[b])."""
    new_k = layer.k.at[phys_block, slot_in_block].set(k_new[:, 0])
    new_v = layer.v.at[phys_block, slot_in_block].set(v_new[:, 0])
    return PagedLayerCache(k=new_k, v=new_v)


def scatter_kv_prefill(
    layer: PagedLayerCache,
    k_new: Array,           # [1, T, H_kv, D] where T is a multiple of block_size
    v_new: Array,
    block_indices: Array,   # [T // block_size] int32
) -> PagedLayerCache:
    """Write a padded prompt of T tokens into the pool via block_indices."""
    T = k_new.shape[1]
    block_size = layer.k.shape[1]
    nb = T // block_size
    k_blocked = k_new[0].reshape(nb, block_size, *k_new.shape[2:])
    v_blocked = v_new[0].reshape(nb, block_size, *v_new.shape[2:])
    new_k = layer.k.at[block_indices].set(k_blocked)
    new_v = layer.v.at[block_indices].set(v_blocked)
    return PagedLayerCache(k=new_k, v=new_v)


def scatter_kv_prefill_batch(
    layer: PagedLayerCache,
    k_new: Array,           # [B, block_size, H_kv, D]
    v_new: Array,
    block_indices: Array,   # [B] int32
) -> PagedLayerCache:
    """Write one fixed-size prefill chunk per batch element."""
    if k_new.shape[1] != layer.k.shape[1]:
        raise ValueError("scatter_kv_prefill_batch expects exactly one block per row")
    new_k = layer.k.at[block_indices].set(k_new)
    new_v = layer.v.at[block_indices].set(v_new)
    return PagedLayerCache(k=new_k, v=new_v)


def gather_kv(
    layer: PagedLayerCache,
    block_tables: Array,    # [B, NB] int32
) -> tuple[Array, Array]:
    """Gather [B, NB*block_size, H_kv, D] K/V views from the pool."""
    k_g = layer.k[block_tables]
    v_g = layer.v[block_tables]
    B, NB, bs, H, D = k_g.shape
    return k_g.reshape(B, NB * bs, H, D), v_g.reshape(B, NB * bs, H, D)
