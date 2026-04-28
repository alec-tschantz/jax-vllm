import hashlib
import struct
from dataclasses import dataclass, field
from typing import Iterable, Optional

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..model.common import KVCacheConfig

BlockHash = bytes
SENTINEL_REF = 1 << 30





class PagedLayerCache(eqx.Module):
    k: Array
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





def compute_block_hash(parent_hash: BlockHash, token_ids: Iterable[int]) -> BlockHash:
    h = hashlib.sha256()
    h.update(parent_hash)
    for tok in token_ids:
        h.update(struct.pack("<I", int(tok)))
    return h.digest()





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
    if num_blocks < 2:
        raise ValueError(f"num_blocks must be >= 2 (1 sentinel + >=1 usable)")
    pool = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
    pool[0].ref_cnt = SENTINEL_REF


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
    block = mgr.pool[block_id]
    if block.ref_cnt == 0:
        _unlink(mgr, block)
    block.ref_cnt += 1


def alloc_new(mgr: BlockManager) -> int:
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
    ids = list(block_ids)
    for block_id in reversed(ids):
        block = mgr.pool[block_id]
        if block.ref_cnt >= SENTINEL_REF:
            continue
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            _append_tail(mgr, block)


def register_hash(mgr: BlockManager, block_id: int, block_hash: BlockHash) -> None:
    block = mgr.pool[block_id]
    if block.block_hash is not None:
        return
    if block_hash in mgr.hash_to_block:
        return
    mgr.hash_to_block[block_hash] = block_id
    block.block_hash = block_hash


def lookup(mgr: BlockManager, block_hash: BlockHash) -> Optional[int]:
    return mgr.hash_to_block.get(block_hash)





def scatter_kv_decode(
    layer: PagedLayerCache,
    k_new: Array,
    v_new: Array,
    phys_block: Array,
    slot_in_block: Array,
) -> PagedLayerCache:
    new_k = layer.k.at[phys_block, slot_in_block].set(k_new[:, 0])
    new_v = layer.v.at[phys_block, slot_in_block].set(v_new[:, 0])
    return PagedLayerCache(k=new_k, v=new_v)


def scatter_kv_prefill(
    layer: PagedLayerCache,
    k_new: Array,
    v_new: Array,
    block_indices: Array,
) -> PagedLayerCache:
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
    k_new: Array,
    v_new: Array,
    block_indices: Array,
) -> PagedLayerCache:
    if k_new.shape[1] != layer.k.shape[1]:
        raise ValueError("scatter_kv_prefill_batch expects exactly one block per row")
    new_k = layer.k.at[block_indices].set(k_new)
    new_v = layer.v.at[block_indices].set(v_new)
    return PagedLayerCache(k=new_k, v=new_v)


def gather_kv(
    layer: PagedLayerCache,
    block_tables: Array,
) -> tuple[Array, Array]:
    k_g = layer.k[block_tables]
    v_g = layer.v[block_tables]
    B, NB, bs, H, D = k_g.shape
    return k_g.reshape(B, NB * bs, H, D), v_g.reshape(B, NB * bs, H, D)
