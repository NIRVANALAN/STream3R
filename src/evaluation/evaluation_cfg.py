from dataclasses import dataclass
from pathlib import Path


@dataclass
class MethodCfg:
    name: str
    key: str
    path: Path


@dataclass
class SceneCfg:
    scene: str
    target_index: int


@dataclass
class EvaluationCfg:
    methods: list[MethodCfg]
    side_by_side_path: Path | None
    output_metrics_path: Path
    animate_side_by_side: bool
    highlighted: list[SceneCfg]
    window_estimator: bool
    eval_cls: str
