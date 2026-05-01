"""
LangGraph orchestrator.

Wires Planner -> Executor -> Critic with a bounded active-perception loop:

START -> planner -> executor -> critic -> output
                              -> correction -> executor
                              -> abstain
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

import critic
import executor
import planner
from state import AgentState, CriticEvidence, SpatialEvidenceGraph

logger = logging.getLogger(__name__)


def planner_node(state: dict, config: RunnableConfig) -> dict:
    s = AgentState(**state)
    llm = config["configurable"]["llm"]
    strict = bool(config["configurable"].get("strict_models", False))
    s = planner.run_planner(s, llm, strict=strict)
    return s.model_dump()


def executor_node(state: dict, config: RunnableConfig) -> dict:
    s = AgentState(**state)
    exec_cfg = config["configurable"].get("executor", {})
    if config["configurable"].get("strict_models", False):
        exec_cfg = {**exec_cfg, "strict": True}
    s = executor.run_executor(s, exec_cfg)
    return s.model_dump()


def critic_node(state: dict, config: RunnableConfig) -> dict:
    s = AgentState(**state)
    critic_cfg = config["configurable"].get("critic", {})
    s = critic.run_critic(s, critic_cfg)
    return s.model_dump()


def correction_node(state: dict, config: RunnableConfig) -> dict:
    """
    Increment iteration counter; the next executor call will use
    state.current_crop, already computed by the Critic.
    """
    s = AgentState(**state)
    s.iteration += 1
    logger.info(f"[Correction] Advancing to iteration {s.iteration}")
    return s.model_dump()


def _latest_evidence(s: AgentState) -> CriticEvidence | None:
    return s.critic_evidence[-1] if s.critic_evidence else None


def _has_geometric_evidence(s: AgentState) -> bool:
    ev = _latest_evidence(s)
    if ev is None:
        return False
    if ev.obj1_bbox is None or ev.obj2_bbox is None:
        return False
    if ev.rule_applied.startswith("unknown relation"):
        return False
    return bool(ev.rule_applied)


def _executor_agrees_with_geometry(s: AgentState) -> bool:
    if not _has_geometric_evidence(s):
        return False
    if s.executor_answer is None:
        return True
    return bool(s.executor_answer) == bool(s.critic_passed)


def output_node(state: dict, config: RunnableConfig) -> dict:
    """Build the final Spatial Evidence Graph from geometric evidence."""
    s = AgentState(**state)

    answer = bool(s.critic_passed)
    answer_str = "yes" if answer else "no"
    geometry_only = s.executor_answer is None

    graph = SpatialEvidenceGraph(
        question=s.question,
        obj1=s.obj1,
        obj2=s.obj2,
        relation=s.relation,
        answer=answer,
        answer_str=answer_str,
        confidence=0.7 if geometry_only else 0.9,
        evidence=s.critic_evidence,
        iterations=s.iteration + 1,
        crop_history=s.crop_history,
        verified=True,
        failure_mode="",
    )
    s.graph = graph
    s.done = True
    logger.info(f"[Output] Final answer: {answer_str} (verified=True)")
    return s.model_dump()


def abstain_node(state: dict, config: RunnableConfig) -> dict:
    """Called when parsing fails or max correction iterations are reached."""
    s = AgentState(**state)

    failure = "unknown"
    if s.error.startswith("Planner error"):
        failure = "planner_parse_error"
    elif s.critic_evidence:
        ev = s.critic_evidence[-1]
        if ev.failure_reason.startswith("detector_miss"):
            failure = "detector_miss"
        elif "depth" in ev.failure_reason:
            failure = "depth_noise"
        elif _has_geometric_evidence(s):
            failure = "vlm_bias"

    graph = SpatialEvidenceGraph(
        question=s.question,
        obj1=s.obj1,
        obj2=s.obj2,
        relation=s.relation,
        answer=None,
        answer_str="abstain",
        confidence=0.0,
        evidence=s.critic_evidence,
        iterations=s.iteration + 1,
        crop_history=s.crop_history,
        verified=False,
        failure_mode=failure,
    )
    s.graph = graph
    s.done = True
    s.abstain = True
    logger.warning(f"[Abstain] {failure}")
    return s.model_dump()


def route_after_planner(state: dict) -> str:
    s = AgentState(**state)
    if s.error.startswith("Planner error") and not (s.obj1 and s.obj2 and s.relation):
        return "abstain"
    return "executor"


def route_after_critic(state: dict) -> str:
    s = AgentState(**state)
    if _executor_agrees_with_geometry(s):
        return "output"
    if s.iteration >= s.max_iterations - 1:
        return "abstain"
    return "correction"


def build_graph() -> StateGraph:
    g = StateGraph(dict)

    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node)
    g.add_node("correction", correction_node)
    g.add_node("output", output_node)
    g.add_node("abstain", abstain_node)

    g.set_entry_point("planner")
    g.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "executor": "executor",
            "abstain": "abstain",
        },
    )
    g.add_edge("executor", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "output": "output",
            "correction": "correction",
            "abstain": "abstain",
        },
    )

    g.add_edge("correction", "executor")
    g.add_edge("output", END)
    g.add_edge("abstain", END)

    return g.compile()


def run_pipeline(
    image_path: str,
    question: str,
    llm: Any,
    executor_config: dict | None = None,
    critic_config: dict | None = None,
    max_iterations: int = 3,
    strict_models: bool = False,
) -> SpatialEvidenceGraph:
    """
    High-level entry point.

    Args:
        image_path: Path to the RGB image.
        question: Binary spatial question in English.
        llm: LangChain ChatModel for the Planner.
        executor_config: dict with executor backend settings.
        critic_config: dict with detector/depth fallback settings.
        max_iterations: Max executor/critic attempts, in {1, 2, 3}.
        strict_models: If true, Planner/Executor model failures raise instead
            of falling back to parser regex or geometry-only behavior.
    """
    pipeline = build_graph()

    init_state = AgentState(
        image_path=str(image_path),
        question=question,
        max_iterations=max_iterations,
    ).model_dump()

    config = {
        "configurable": {
            "llm": llm,
            "executor": executor_config or {"backend": "local"},
            "critic": critic_config or {},
            "strict_models": strict_models,
        }
    }

    result = pipeline.invoke(init_state, config=config)
    final = AgentState(**result)
    return final.graph
