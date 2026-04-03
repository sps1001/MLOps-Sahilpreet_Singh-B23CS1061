from __future__ import annotations

from typing import Any, Dict, Optional


def init_wandb(
    *,
    enabled: bool,
    entity: str,
    project: str,
    name: Optional[str],
    config: Dict[str, Any],
):
    if not enabled:
        return None

    import wandb

    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config,
    )
    return run

