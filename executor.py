"""
Executor Agent - vision model wrapper.

Produces an initial yes/no hypothesis and two object claims that the Critic
will localize. Supports local LLaVA, OpenAI-compatible vision APIs, and a
deterministic mock backend for smoke tests and demos.
"""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import Any, Optional

from PIL import Image
from pydantic import BaseModel, Field, field_validator

from parsing import extract_json_object

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM = """\
You are a Visual Hypothesis Generator. Your job is to examine an image and a
spatial question, then output a structured JSON record.

Output ONLY valid JSON - no markdown, no explanation:
{
  "answer": "yes" | "no",
  "confidence": <float 0-1>,
  "claims": [
    {
      "obj": "<object noun phrase as it appears in the image>",
      "role": "subject" | "reference",
      "visible": true | false
    }
  ],
  "rationale": "<one sentence>"
}

Rules:
- "answer" is your best yes/no guess before geometric verification.
- List exactly the two objects from the question as claims (subject first).
- Use the most specific visible noun phrase (e.g., "red ceramic mug" not "cup").
- Be honest about visibility; set visible=false if the object is occluded/absent.
"""


class ExecutorClaim(BaseModel):
    obj: str = ""
    role: str = ""
    visible: bool = True

    @field_validator("obj", "role")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return str(value).strip()

    @field_validator("role")
    @classmethod
    def _normalize_role(cls, value: str) -> str:
        role = value.lower().strip()
        if role not in {"subject", "reference"}:
            raise ValueError(f"unsupported claim role: {value!r}")
        return role


class ExecutorOutput(BaseModel):
    answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    claims: list[ExecutorClaim] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        text = str(value).strip().lower().strip(".! ")
        if text in {"yes", "true", "1"}:
            return "yes"
        if text in {"no", "false", "0"}:
            return "no"
        raise ValueError(f"unsupported answer: {value!r}")


def _image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _load_image(image_path: str, crop=None) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    if crop is not None:
        w, h = img.size
        x1 = int(crop.x1 * w)
        y1 = int(crop.y1 * h)
        x2 = int(crop.x2 * w)
        y2 = int(crop.y2 * h)
        img = img.crop((x1, y1, x2, y2))
    return img


_llava_model = None
_llava_processor = None


def _load_llava(model_id: str = "llava-hf/llava-1.5-7b-hf", load_in_4bit: bool = True) -> None:
    global _llava_model, _llava_processor
    if _llava_model is not None:
        return

    import torch
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    logger.info(f"[Executor] Loading LLaVA from {model_id}")
    _llava_processor = LlavaProcessor.from_pretrained(model_id)
    _llava_model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )
    logger.info("[Executor] LLaVA loaded.")


def _run_llava_local(
    image: Image.Image,
    question: str,
    obj1: str,
    obj2: str,
    relation: str,
    config: dict,
) -> str:
    import torch

    _load_llava(
        model_id=config.get("model_id", "llava-hf/llava-1.5-7b-hf"),
        load_in_4bit=bool(config.get("load_in_4bit", True)),
    )

    prompt_text = (
        "<image>\n"
        f"USER: {EXECUTOR_SYSTEM}\n\n"
        f"Question: Is the {obj1} {relation.replace('_', ' ')} the {obj2}?\n"
        "ASSISTANT:"
    )

    inputs = _llava_processor(
        text=prompt_text,
        images=image,
        return_tensors="pt",
    ).to(_llava_model.device, torch.float16)

    with torch.no_grad():
        output_ids = _llava_model.generate(
            **inputs,
            max_new_tokens=int(config.get("max_new_tokens", 300)),
            do_sample=False,
        )

    generated = _llava_processor.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in generated:
        generated = generated.split("ASSISTANT:", 1)[1].strip()
    return generated


def _run_openai_vision(
    image: Image.Image,
    question: str,
    obj1: str,
    obj2: str,
    relation: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: str = "gpt-4o",
) -> str:
    import requests

    url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
    b64 = _image_to_base64(image)
    payload = {
        "model": model,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": EXECUTOR_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {
                        "type": "text",
                        "text": f"Question: Is the {obj1} {relation.replace('_', ' ')} the {obj2}?",
                    },
                ],
            },
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _run_mock_executor(obj1: str, obj2: str, answer: str = "yes") -> str:
    answer = "yes" if str(answer).strip().lower() in {"yes", "true", "1"} else "no"
    return (
        '{"answer": "%s", "confidence": 0.8, '
        '"claims": ['
        '{"obj": "%s", "role": "subject", "visible": true}, '
        '{"obj": "%s", "role": "reference", "visible": true}'
        '], "rationale": "mock executor response"}'
    ) % (answer, obj1.replace('"', "'"), obj2.replace('"', "'"))


def _parse_executor_output(raw: str) -> ExecutorOutput:
    return ExecutorOutput.model_validate(extract_json_object(raw))


def _claims_from_output(parsed: ExecutorOutput, obj1: str, obj2: str) -> list[str]:
    role_claims: dict[str, str] = {}
    ordered_visible: list[str] = []

    for claim in parsed.claims:
        if not claim.visible or not claim.obj:
            continue
        ordered_visible.append(claim.obj)
        role_claims[claim.role] = claim.obj

    if role_claims.get("subject") and role_claims.get("reference"):
        return [role_claims["subject"], role_claims["reference"]]
    if len(ordered_visible) >= 2:
        return ordered_visible[:2]
    return [obj1, obj2]


def run_executor(state: Any, config: dict) -> Any:
    """
    Executor node.

    config keys:
      backend       : "local" | "openai" | "mock"  (default "local")
      openai_key    : str (required if backend=="openai")
      openai_base   : str (optional, for custom endpoint)
      model         : str (optional OpenAI-compatible vision model)
      mock_answer   : "yes" | "no" (for backend=="mock")
    """
    backend = config.get("backend", "local")
    strict = bool(config.get("strict", False))
    crop = state.current_crop

    logger.info(f"[Executor] iter={state.iteration}, backend={backend}, crop={crop}")

    image = _load_image(state.image_path, crop)

    try:
        if backend == "openai":
            api_key = config.get("openai_key") or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            raw = _run_openai_vision(
                image,
                state.question,
                state.obj1,
                state.obj2,
                state.relation,
                api_key=api_key,
                base_url=config.get("openai_base"),
                model=config.get("model", "gpt-4o"),
            )
        elif backend == "mock":
            raw = _run_mock_executor(state.obj1, state.obj2, config.get("mock_answer", "yes"))
        else:
            raw = _run_llava_local(image, state.question, state.obj1, state.obj2, state.relation, config)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Executor backend failed: {exc}") from exc
        logger.warning(f"[Executor] backend failed ({exc}); using planner nouns for critic")
        state.error = f"Executor backend error: {exc}"
        state.executor_raw = ""
        state.executor_answer = None
        state.executor_claims = [state.obj1, state.obj2]
        return state

    state.executor_raw = raw
    logger.debug(f"[Executor] raw output:\n{raw}")

    try:
        parsed = _parse_executor_output(raw)
        state.executor_answer = parsed.answer == "yes"
        state.executor_claims = _claims_from_output(parsed, state.obj1, state.obj2)
        state.error = ""
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Executor output parse failed: {exc}") from exc
        logger.warning(f"[Executor] parse failed ({exc}); using planner nouns for critic")
        state.error = f"Executor parse error: {exc}"
        state.executor_answer = None
        state.executor_claims = [state.obj1, state.obj2]

    logger.info(
        f"[Executor] answer={state.executor_answer}, claims={state.executor_claims}"
    )
    return state
