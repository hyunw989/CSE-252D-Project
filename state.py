"""
Spatial Evidence Graph State
Core state object passed between all agents in the pipeline.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box [x1, y1, x2, y2] in [0,1] range."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    label: str = ""

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: "BoundingBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class CriticEvidence(BaseModel):
    """Geometric evidence produced by the Critic for one claim."""
    claim: str = ""
    passed: bool = False
    obj1_bbox: Optional[BoundingBox] = None
    obj2_bbox: Optional[BoundingBox] = None
    obj1_depth: Optional[float] = None
    obj2_depth: Optional[float] = None
    dx: Optional[float] = None      # cx(obj1) - cx(obj2), horizontal
    dy: Optional[float] = None      # cy(obj1) - cy(obj2), vertical
    dz: Optional[float] = None      # depth(obj1) - depth(obj2)
    iou: Optional[float] = None
    rule_applied: str = ""
    failure_reason: str = ""


class CropRegion(BaseModel):
    """Crop coordinates (normalized) used for active-perception fallback."""
    x1: float
    y1: float
    x2: float
    y2: float
    reason: str = ""


class SpatialEvidenceGraph(BaseModel):
    """
    Full Spatial Evidence Graph — the structured output of the pipeline.
    Combines scene-graph structure with explicit geometric constraints.
    """
    question: str = ""
    obj1: str = ""
    obj2: str = ""
    relation: str = ""
    answer: Optional[bool] = None           # True=yes, False=no, None=abstain
    answer_str: str = ""                    # "yes" / "no" / "abstain"
    confidence: float = 0.0
    evidence: list[CriticEvidence] = Field(default_factory=list)
    iterations: int = 0
    crop_history: list[CropRegion] = Field(default_factory=list)
    failure_mode: str = ""                  # "detector_miss" | "depth_noise" | "vlm_bias" | ""
    verified: bool = False


class AgentState(BaseModel):
    """
    LangGraph state — the single object flowing through all nodes.
    """
    # ── Input ─────────────────────────────────────────────────────────
    image_path: str = ""
    question: str = ""

    # ── Planner output ────────────────────────────────────────────────
    obj1: str = ""
    obj2: str = ""
    relation: str = ""          # one of: left_of, right_of, above, below, behind, in_front, on, contains
    planner_raw: str = ""

    # ── Executor output ───────────────────────────────────────────────
    executor_answer: Optional[bool] = None
    executor_claims: list[str] = Field(default_factory=list)
    executor_raw: str = ""

    # ── Critic output ─────────────────────────────────────────────────
    critic_passed: bool = False
    critic_evidence: list[CriticEvidence] = Field(default_factory=list)

    # ── Loop control ──────────────────────────────────────────────────
    iteration: int = 0
    max_iterations: int = 3
    current_crop: Optional[CropRegion] = None
    crop_history: list[CropRegion] = Field(default_factory=list)

    # ── Final output ──────────────────────────────────────────────────
    graph: SpatialEvidenceGraph = Field(default_factory=SpatialEvidenceGraph)
    done: bool = False
    abstain: bool = False
    error: str = ""

    class Config:
        arbitrary_types_allowed = True
