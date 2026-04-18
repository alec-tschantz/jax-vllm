from dataclasses import dataclass

from .paged import (
    BlockHash,
    BlockManager,
    compute_block_hash,
    lookup,
    register_hash,
    release,
    touch,
)


@dataclass(frozen=True)
class CachedPrefix:
    block_ids: list[int]
    hashes: list[BlockHash]


def prompt_block_hashes(prompt_ids: list[int], block_size: int) -> list[BlockHash]:
    """Return full-block prompt hashes; trailing partial blocks are skipped."""
    nb_full = len(prompt_ids) // block_size
    hashes: list[BlockHash] = []
    parent: BlockHash = b""
    for block_idx in range(nb_full):
        start = block_idx * block_size
        end = start + block_size
        block_hash = compute_block_hash(parent, prompt_ids[start:end])
        hashes.append(block_hash)
        parent = block_hash
    return hashes


def take_cached_prefix(
    mgr: BlockManager, prompt_ids: list[int], block_size: int
) -> CachedPrefix:
    hashes = prompt_block_hashes(prompt_ids, block_size)
    block_ids: list[int] = []
    cached_hashes: list[BlockHash] = []
    for block_hash in hashes:
        block_id = lookup(mgr, block_hash)
        if block_id is None:
            break
        touch(mgr, block_id)
        block_ids.append(block_id)
        cached_hashes.append(block_hash)
    return CachedPrefix(block_ids=block_ids, hashes=cached_hashes)


def register_full_block(
    mgr: BlockManager,
    block_id: int,
    parent_hash: BlockHash,
    token_ids: list[int],
) -> BlockHash:
    block_hash = compute_block_hash(parent_hash, token_ids)
    register_hash(mgr, block_id, block_hash)
    return block_hash


def release_blocks(mgr: BlockManager, block_ids: list[int]) -> None:
    release(mgr, block_ids)
