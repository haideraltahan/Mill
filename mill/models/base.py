"""Shared utilities for Mill model backends."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mill.api.instance import Instance

logger = logging.getLogger(__name__)


def collate_by_gen_kwargs(requests: list["Instance"]) -> list[list["Instance"]]:
    """Group requests that share the same generation kwargs so they can be batched."""
    from collections import defaultdict
    import json

    groups: dict[str, list] = defaultdict(list)
    for req in requests:
        gen_kwargs = req.arguments[1] if len(req.arguments) > 1 else {}
        key = json.dumps(gen_kwargs, sort_keys=True, default=str)
        groups[key].append(req)
    return list(groups.values())


def chunk(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def is_multimodal_request(request: "Instance") -> bool:
    from mill.api.protocol import ChatMessages
    return isinstance(request.arguments[0], ChatMessages)
