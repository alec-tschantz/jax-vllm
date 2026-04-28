import jax.numpy as jnp
import numpy as np

from jllm.engine.paged import (
    PagedLayerCache,
    alloc_new,
    compute_block_hash,
    gather_kv,
    lookup,
    make_manager,
    register_hash,
    release,
    scatter_kv_decode,
    scatter_kv_prefill,
    touch,
)


# ---------- device-side primitives ----------


def _empty_layer(num_blocks, block_size, H=2, D=4):
    shape = (num_blocks, block_size, H, D)
    return PagedLayerCache(k=jnp.zeros(shape, jnp.float32), v=jnp.zeros(shape, jnp.float32))


def test_scatter_decode_then_gather():
    layer = _empty_layer(8, 4, 2, 4)
    k_new = jnp.stack([
        jnp.full((1, 2, 4), 1.0, dtype=jnp.float32),
        jnp.full((1, 2, 4), 2.0, dtype=jnp.float32),
    ])
    v_new = k_new * 10.0
    phys_block = jnp.asarray([3, 5], jnp.int32)
    slot_in_block = jnp.asarray([2, 0], jnp.int32)
    new_layer = scatter_kv_decode(layer, k_new, v_new, phys_block, slot_in_block)

    tables = jnp.asarray([[3, 0], [5, 0]], jnp.int32)
    k_flat, v_flat = gather_kv(new_layer, tables)
    assert k_flat.shape == (2, 8, 2, 4)
    assert float(k_flat[0, 2, 0, 0]) == 1.0
    assert float(v_flat[0, 2, 0, 0]) == 10.0
    assert float(k_flat[1, 0, 0, 0]) == 2.0
    assert float(v_flat[1, 0, 0, 0]) == 20.0
    assert float(k_flat[0, 0, 0, 0]) == 0.0
    assert float(k_flat[1, 2, 0, 0]) == 0.0


def test_scatter_prefill_then_gather():
    layer = _empty_layer(8, 4, 2, 4)
    T = 8
    k_new = jnp.arange(T * 2 * 4, dtype=jnp.float32).reshape(1, T, 2, 4)
    v_new = k_new + 100.0
    block_indices = jnp.asarray([3, 5], jnp.int32)
    new_layer = scatter_kv_prefill(layer, k_new, v_new, block_indices)

    tables = jnp.asarray([[3, 5]], jnp.int32)
    k_flat, v_flat = gather_kv(new_layer, tables)
    assert k_flat.shape == (1, 8, 2, 4)
    np.testing.assert_array_equal(np.asarray(k_flat[0]), np.asarray(k_new[0]))
    np.testing.assert_array_equal(np.asarray(v_flat[0]), np.asarray(v_new[0]))


def test_gather_reads_only_listed_blocks():
    layer = _empty_layer(8, 4, 2, 4)
    layer = scatter_kv_decode(
        layer,
        jnp.full((2, 1, 2, 4), 7.0, jnp.float32),
        jnp.full((2, 1, 2, 4), 8.0, jnp.float32),
        jnp.asarray([0, 4], jnp.int32),
        jnp.asarray([0, 0], jnp.int32),
    )
    tables = jnp.asarray([[0, 1], [4, 2]], jnp.int32)
    k_flat, _ = gather_kv(layer, tables)
    assert float(k_flat[0, 0, 0, 0]) == 7.0
    assert float(k_flat[1, 0, 0, 0]) == 7.0
    for b, pos in [(0, 4), (1, 4)]:
        assert float(k_flat[b, pos, 0, 0]) == 0.0


# ---------- BlockManager: ref counting + LRU + hash map ----------


def test_manager_alloc_releases_round_trip():
    """alloc_new pops from free head; release appends to free tail in reverse."""
    mgr = make_manager(8)
    # Block 0 is sentinel. Free queue: 1..7.
    a = alloc_new(mgr)
    b = alloc_new(mgr)
    c = alloc_new(mgr)
    assert a == 1 and b == 2 and c == 3
    assert mgr.pool[a].ref_cnt == 1

    # Release in ascending order; reverse-append means c,b,a appended to tail.
    release(mgr, [a, b, c])
    assert mgr.pool[a].ref_cnt == 0
    # Free queue order after release: [4,5,6,7,c,b,a]  (c=3 first appended, then b=2, then a=1)
    order = []
    cur = mgr.free_head
    while cur is not None:
        order.append(cur.block_id)
        cur = cur.next_free
    assert order == [4, 5, 6, 7, 3, 2, 1]


def test_manager_sentinel_never_in_free_queue():
    """Block 0 has SENTINEL_REF and must never appear in the free queue."""
    mgr = make_manager(4)
    # Walk the whole free queue twice: block 0 must not appear.
    cur = mgr.free_head
    seen = []
    while cur is not None:
        seen.append(cur.block_id)
        cur = cur.next_free
    assert 0 not in seen
    # Release with block 0 in the list: should skip it.
    release(mgr, [0])
    cur = mgr.free_head
    while cur is not None:
        assert cur.block_id != 0
        cur = cur.next_free


def test_manager_touch_unlinks_cached_block():
    """touch() on a block with ref_cnt=0 unlinks it from the free queue; another
    touch just increments ref_cnt."""
    mgr = make_manager(4)
    # Register a fake hash for block 2 (simulate a freshly-cached block that
    # hasn't been reused yet — it's in the free queue with block_hash set).
    # We do this by allocating, registering, and releasing.
    b = alloc_new(mgr)
    register_hash(mgr, b, b"fake-hash")
    release(mgr, [b])
    assert mgr.pool[b].ref_cnt == 0
    assert mgr.pool[b].block_hash == b"fake-hash"
    # touch: should unlink and set ref_cnt=1
    touch(mgr, b)
    assert mgr.pool[b].ref_cnt == 1
    # Free queue must not contain b any more.
    cur = mgr.free_head
    seen = []
    while cur is not None:
        seen.append(cur.block_id)
        cur = cur.next_free
    assert b not in seen
    # Second touch: ref_cnt 2, still not in free queue.
    touch(mgr, b)
    assert mgr.pool[b].ref_cnt == 2


def test_manager_alloc_evicts_cached_hash():
    """If the popped free-head block has a block_hash, the hash map entry is
    removed on alloc (eviction)."""
    mgr = make_manager(4)
    b = alloc_new(mgr)
    register_hash(mgr, b, b"hash-A")
    assert lookup(mgr, b"hash-A") == b
    release(mgr, [b])
    # Now b is at the free queue TAIL (last released). To force eviction, we
    # need to exhaust the rest of the queue so b becomes the head. Alloc twice.
    # Free queue after release: [heads of 4-block pool minus b, then b at tail].
    # We had blocks 1,2,3 free initially; after alloc b=1 and release back, b
    # goes to tail, so free queue = [2,3,1] (head=2).
    a1 = alloc_new(mgr)  # pops 2
    a2 = alloc_new(mgr)  # pops 3
    a3 = alloc_new(mgr)  # pops 1 (== b); its hash gets evicted
    assert a3 == b
    assert lookup(mgr, b"hash-A") is None
    assert mgr.pool[b].block_hash is None


def test_manager_register_hash_tolerates_duplicates():
    """setdefault semantics: if the hash is already mapped to another block,
    the new block is NOT overwritten (v1 duplicate-tolerance rule)."""
    mgr = make_manager(8)
    b1 = alloc_new(mgr)
    b2 = alloc_new(mgr)
    register_hash(mgr, b1, b"hash-X")
    register_hash(mgr, b2, b"hash-X")  # duplicate; should not replace
    assert lookup(mgr, b"hash-X") == b1
    assert mgr.pool[b1].block_hash == b"hash-X"
    assert mgr.pool[b2].block_hash is None


def test_hash_merkle_chain_and_determinism():
    """Hash depends on parent + tokens; deterministic across calls."""
    h0 = compute_block_hash(b"", [1, 2, 3, 4])
    h0_again = compute_block_hash(b"", [1, 2, 3, 4])
    assert h0 == h0_again
    # Different tokens -> different hash
    h_diff = compute_block_hash(b"", [1, 2, 3, 5])
    assert h0 != h_diff
    # Different parent -> different hash
    h_child = compute_block_hash(h0, [5, 6, 7, 8])
    h_child_diff_parent = compute_block_hash(h_diff, [5, 6, 7, 8])
    assert h_child != h_child_diff_parent
    # Same parent + same tokens -> same hash (full determinism)
    h_child_again = compute_block_hash(h0, [5, 6, 7, 8])
    assert h_child == h_child_again
