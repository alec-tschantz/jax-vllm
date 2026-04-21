from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    eos_id: Optional[int] = None


@dataclass
class Request:
    request_id: int
    prompt_ids: list[int]
    sampling: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    finished: bool = False
    slot: Optional[int] = None


@dataclass
class StepEvent:
    request_id: int
    token: int
    finished: bool


@dataclass
class EngineStats:
    admissions: int = 0
    scheduler_loops: int = 0
    prefill_batches: int = 0
    decode_batches: int = 0
    prefill_slots_total: int = 0
    decode_slots_total: int = 0
    prefill_padding_slots_total: int = 0
    decode_padding_slots_total: int = 0
    extend_calls: int = 0
    decode_calls: int = 0
    emitted_tokens: int = 0
    cache_hit_blocks: int = 0
