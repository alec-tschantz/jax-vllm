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
