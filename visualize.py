"""
Visualization utilities.
"""

from __future__ import annotations

import json
from typing import Optional

from PIL import Image, ImageDraw


def print_graph(graph, verbose: bool = True) -> None:
    """Pretty-print the SpatialEvidenceGraph to stdout."""
    reset = "\033[0m"
    bold = "\033[1m"
    green = "\033[32m"
    red = "\033[31m"
    yellow = "\033[33m"
    cyan = "\033[36m"

    colour = green if graph.verified else (yellow if graph.answer_str == "abstain" else red)

    print(f"\n{'=' * 60}")
    print(f"{bold}  Spatial Evidence Graph{reset}")
    print(f"{'=' * 60}")
    print(f"  Question  : {graph.question}")
    print(f"  Predicate : {graph.obj1!r}  [{graph.relation.replace('_', ' ')}]  {graph.obj2!r}")
    print(
        f"  Answer    : {colour}{bold}{graph.answer_str.upper()}{reset}  "
        f"(confidence={graph.confidence:.2f})"
    )
    print(f"  Verified  : {'yes' if graph.verified else 'no'}  Iterations: {graph.iterations}")

    if graph.failure_mode:
        print(f"  Failure   : {red}{graph.failure_mode}{reset}")

    if verbose and graph.evidence:
        print(f"\n  {cyan}Evidence:{reset}")
        for index, ev in enumerate(graph.evidence, 1):
            status = f"{green}TRUE{reset}" if ev.passed else f"{yellow}FALSE{reset}"
            print(f"    [{index}] {ev.claim} -> {status}")
            print(f"        Rule   : {ev.rule_applied}")
            if ev.obj1_bbox:
                b = ev.obj1_bbox
                print(
                    "        obj1   : "
                    f"cx={b.cx:.3f} cy={b.cy:.3f} "
                    f"box=[{b.x1:.2f},{b.y1:.2f},{b.x2:.2f},{b.y2:.2f}] "
                    f"conf={b.confidence:.2f}"
                )
            if ev.obj2_bbox:
                b = ev.obj2_bbox
                print(
                    "        obj2   : "
                    f"cx={b.cx:.3f} cy={b.cy:.3f} "
                    f"box=[{b.x1:.2f},{b.y1:.2f},{b.x2:.2f},{b.y2:.2f}] "
                    f"conf={b.confidence:.2f}"
                )
            if ev.obj1_depth is not None:
                print(f"        depth  : obj1={ev.obj1_depth:.4f} obj2={ev.obj2_depth:.4f} dz={ev.dz:.4f}")
            if ev.failure_reason and not graph.verified:
                print(f"        Reason : {red}{ev.failure_reason}{reset}")

    if verbose and graph.crop_history:
        print(f"\n  {cyan}Crop history:{reset}")
        for index, crop in enumerate(graph.crop_history, 1):
            print(
                f"    [{index}] "
                f"[{crop.x1:.2f},{crop.y1:.2f},{crop.x2:.2f},{crop.y2:.2f}] "
                f"{crop.reason}"
            )

    print(f"{'=' * 60}\n")


def annotate_image(
    image_path: str,
    graph,
    out_path: Optional[str] = None,
) -> Image.Image:
    """
    Draw bounding boxes, centroids, and a relation line on the original image.
    Evidence boxes are stored in original-image normalized coordinates.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    colors = {
        "obj1": (52, 211, 153),
        "obj2": (251, 113, 133),
        "arrow": (250, 204, 21),
    }

    def _px(nx: float, ny: float) -> tuple[int, int]:
        return int(nx * w), int(ny * h)

    for ev in graph.evidence:
        for obj_key, bbox, color_key in [
            (graph.obj1, ev.obj1_bbox, "obj1"),
            (graph.obj2, ev.obj2_bbox, "obj2"),
        ]:
            if bbox is None:
                continue
            tl = _px(bbox.x1, bbox.y1)
            br = _px(bbox.x2, bbox.y2)
            draw.rectangle([tl, br], outline=colors[color_key], width=3)
            draw.text((tl[0] + 4, tl[1] + 4), obj_key, fill=colors[color_key])

        if ev.obj1_bbox and ev.obj2_bbox:
            c1 = _px(ev.obj1_bbox.cx, ev.obj1_bbox.cy)
            c2 = _px(ev.obj2_bbox.cx, ev.obj2_bbox.cy)
            draw.line([c1, c2], fill=colors["arrow"], width=2)
            draw.ellipse([c1[0] - 4, c1[1] - 4, c1[0] + 4, c1[1] + 4], fill=colors["arrow"])
            draw.ellipse([c2[0] - 4, c2[1] - 4, c2[0] + 4, c2[1] + 4], fill=colors["arrow"])

    verdict_color = (52, 211, 153) if graph.verified else (251, 113, 133)
    draw.text(
        (8, 8),
        f"{graph.answer_str.upper()} ({graph.relation.replace('_', ' ')})",
        fill=verdict_color,
    )

    if out_path:
        img.save(out_path)
        print(f"[Viz] Saved annotated image -> {out_path}")

    return img


def export_graph_json(graph, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(graph.model_dump(), handle, indent=2, default=str)
    print(f"[Export] Evidence graph saved -> {out_path}")
