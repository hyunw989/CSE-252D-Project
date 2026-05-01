"""
Planner Agent
Parses a natural-language spatial question into (obj1, obj2, relation).
Uses an LLM with a strict system prompt to extract structured output.
Falls back to a conservative grammar when the LLM response cannot be parsed.
"""

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from parsing import extract_json_object

logger = logging.getLogger(__name__)

CANONICAL_RELATIONS = {
    "left_of",
    "right_of",
    "above",
    "below",
    "behind",
    "in_front",
    "on",
    "contains",
}

RelationName = Literal[
    "left_of",
    "right_of",
    "above",
    "below",
    "behind",
    "in_front",
    "on",
    "contains",
]

RELATION_ALIASES = {
    "left": "left_of",
    "left of": "left_of",
    "to the left of": "left_of",
    "right": "right_of",
    "right of": "right_of",
    "to the right of": "right_of",
    "under": "below",
    "beneath": "below",
    "underneath": "below",
    "over": "above",
    "on top": "on",
    "on top of": "on",
    "contain": "contains",
    "contains": "contains",
    "containing": "contains",
    "in front of": "in_front",
    "front of": "in_front",
}

PLANNER_SYSTEM_PROMPT = """\
You are a Spatial Query Planner. Given a yes/no question about a spatial relationship
between two objects in an image, extract exactly:

  obj1   : the first (subject) object, as a concise noun phrase
  obj2   : the second (reference) object, as a concise noun phrase
  relation: one of [left_of, right_of, above, below, behind, in_front, on, contains]

Rules:
- The question always asks whether obj1 <relation> obj2.
- Use ONLY one of the canonical relation names listed above.
- "behind" means further from the camera (greater depth).
- "contains" means obj1 spatially encloses obj2 (e.g., bowl contains apple).
- For "inside" or "within" questions, rewrite as contains by swapping the objects
  (e.g., "apple inside bowl" becomes obj1="bowl", relation="contains", obj2="apple").
- Output ONLY valid JSON, no commentary, no markdown fences.

Output format:
{"obj1": "...", "obj2": "...", "relation": "..."}
"""


def _normalize_relation(rel: str) -> str:
    rel_text = re.sub(r"[\s_-]+", " ", rel.strip().lower())
    rel_norm = rel_text.replace(" ", "_")
    if rel_norm in CANONICAL_RELATIONS:
        return rel_norm
    for alias, canonical in RELATION_ALIASES.items():
        if alias == rel_text:
            return canonical
    return rel_norm


class PlannerOutput(BaseModel):
    obj1: str
    obj2: str
    relation: RelationName

    @field_validator("obj1", "obj2")
    @classmethod
    def _non_empty_object(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("object name is required")
        return value

    @field_validator("relation", mode="before")
    @classmethod
    def _canonical_relation(cls, value: str) -> str:
        relation = _normalize_relation(str(value))
        if relation not in CANONICAL_RELATIONS:
            raise ValueError(f"unsupported relation: {value!r}")
        return relation


def run_planner(state: Any, llm: Any, strict: bool = False) -> Any:
    """
    Planner node. Calls the LLM and updates state with obj1, obj2, relation.
    `llm` must be a LangChain-compatible chat model.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    question = state.question
    logger.info(f"[Planner] Question: {question}")

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f'Question: "{question}"'),
    ]

    try:
        parsed = _invoke_structured_planner(llm, messages)
        state.obj1 = parsed.obj1
        state.obj2 = parsed.obj2
        state.relation = parsed.relation
        state.error = ""
        logger.info(
            f"[Planner] obj1={state.obj1!r}, obj2={state.obj2!r}, "
            f"rel={state.relation!r}"
        )

    except Exception as exc:
        if strict:
            raise RuntimeError(f"Planner model failed: {exc}") from exc
        logger.warning(f"[Planner] LLM parse failed ({exc}), trying regex fallback")
        if not _regex_fallback(state, question):
            state.error = f"Planner error: {exc}"

    return state


def _invoke_structured_planner(llm: Any, messages: list[Any]) -> PlannerOutput:
    """Use LangChain structured output when available; fall back to JSON text."""
    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(PlannerOutput)
        except (AttributeError, NotImplementedError, TypeError, ValueError) as exc:
            logger.debug(f"[Planner] structured output path failed: {exc}")
        else:
            result = structured_llm.invoke(messages)
            if isinstance(result, PlannerOutput):
                return result
            return PlannerOutput.model_validate(result)

    response = llm.invoke(messages)
    raw = response.content.strip()
    return PlannerOutput.model_validate(extract_json_object(raw))


def _clean_object_phrase(text: str) -> str:
    text = text.strip(" \t\r\n\"'")
    text = re.sub(r"^(is|are|was|were|do|does|did|can|could|would|will)\s+", "", text)
    text = re.sub(r"^there\s+(is|are)\s+", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text.strip()


def _regex_fallback(state: Any, question: str) -> bool:
    """
    Conservative regex fallback when LLM output cannot be parsed.
    Handles patterns like "Is the X [relation] the Y?"
    """
    q = question.lower().strip().rstrip("?")

    relation_patterns = [
        (r"\b(to\s+the\s+left\s+of|left\s+of)\b", "left_of", False),
        (r"\b(to\s+the\s+right\s+of|right\s+of)\b", "right_of", False),
        (r"\bin\s+front\s+of\b", "in_front", False),
        (r"\bbehind\b", "behind", False),
        (r"\b(above|over)\b", "above", False),
        (r"\b(below|under|beneath|underneath)\b", "below", False),
        (r"\bon\s+top\s+of\b|\bon\b", "on", False),
        (r"\b(contains?|containing|holds?|holding)\b", "contains", False),
        (r"\b(inside|within)\b", "contains", True),
    ]

    for pattern, relation, swap_objects in relation_patterns:
        match = re.search(pattern, q)
        if match is None:
            continue

        before = q[: match.start()].strip()
        after = q[match.end() :].strip()
        obj1 = _clean_object_phrase(before)
        obj2 = _clean_object_phrase(after)
        if swap_objects:
            obj1, obj2 = obj2, obj1

        if not obj1 or not obj2:
            continue

        state.obj1 = obj1
        state.obj2 = obj2
        state.relation = relation
        state.error = ""
        logger.info(
            f"[Planner] regex fallback obj1={state.obj1!r}, "
            f"obj2={state.obj2!r}, rel={state.relation!r}"
        )
        return True

    return False
