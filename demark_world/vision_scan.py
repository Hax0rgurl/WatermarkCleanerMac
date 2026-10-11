"""Local vision scanning and conversion of observations into editable ranges."""

import json
import math
import statistics
import uuid
from urllib.parse import urlparse


VISION_MODELS = {
    "Latest quality": ("mlx-community/Qwen3.8-27B-4bit",
                       "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"),
    "Quick": ("mlx-community/Qwen3.5-4B-4bit",
              "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"),
}

WATERMARK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["watermarks"],
    "properties": {"watermarks": {"type": "array", "maxItems": 12,
        "items": {"type": "object", "additionalProperties": False,
                  "required": ["label", "bbox"], "properties": {
            "label": {"type": "string", "maxLength": 100},
            "bbox": {"type": "array", "minItems": 4, "maxItems": 4,
                     "items": {"type": "integer", "minimum": 0, "maximum": 1000}},
        }}}},
}

PROMPT = '''IMPORTANT: Distinguish editorial overlays from ownership/platform branding.
Numbered topic headings, section/chapter titles, one-word advice or instructions are ordinary
captions, even when they appear in a corner. Capital letters and screen position alone do
not make text a watermark. Only return an overlay if it identifies the platform, creator,
owner or software. If the only overlay is a numbered heading or an instructional word,
return an empty watermarks list.
Find visible overlaid WATERMARKS or creator/platform branding in this video frame.
Include logos, small brand names, usernames, translucent brand text, and tiled watermarks.
Exclude subtitles, ordinary captions, titles, explanatory text, and text on objects in the scene.
Do not invent a watermark if none is visible. A platform logo and its adjacent creator name are one watermark.
Return ONLY JSON: {"watermarks":[{"label":"brand or description",
"bbox":[left,top,right,bottom]}]}. Coordinates MUST be integers normalized to 0..1000
relative to the entire image. Include the complete branding group, its text AND adjacent
graphic icon/emblem, in one box. Do not return only lettering when there is an adjacent icon.
Return {"watermarks":[]} if there are no watermarks.'''


def parse_watermarks(text: str) -> list[dict]:
    # Reject truncated/malformed answers instead of treating them as absent logos.
    text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("Vision model did not return a valid watermark result.")
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    items = payload.get("watermarks")
    if not isinstance(items, list):
        raise ValueError("Vision model returned an invalid watermark list.")
    result = []
    for item in items:
        box = item.get("bbox") if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Vision model returned invalid box coordinates.")
        box = [float(v) / 1000 for v in box]
        if not all(math.isfinite(v) for v in box) or not (
            0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
        ):
            raise ValueError("Vision model returned a box outside the image.")
        result.append({"bbox": box, "label": str(item.get("label", "Watermark"))[:100]})
    return result


def box_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union if union else 0


def sample_indices(total_frames: int, fps: float, seconds: float) -> list[int]:
    if total_frames < 1 or not math.isfinite(fps) or fps <= 0 or seconds <= 0:
        raise ValueError("Invalid video or sampling interval.")
    step = max(1, round(seconds * fps))
    return sorted(set(range(0, total_frames, step)) | {total_frames - 1})


def ranges_from_observations(observations: list[dict], total_frames: int) -> list[dict]:
    """Associate simultaneous boxes, splitting every observed disappearance.

    Boundaries fall halfway between samples. No interpolation across negative samples.
    Suggested ranges remain approximate and must be reviewed by the user.
    """
    tracks, active = [], []
    for sample, observation in enumerate(observations):
        available = list(active)
        next_active = []
        for detection in observation["watermarks"]:
            matches = [(box_iou(track["last_box"], detection["bbox"]), track)
                       for track in available]
            score, track = max(matches, key=lambda pair: pair[0]) if matches else (0, None)
            if score < .35:
                track = {"first": sample, "last": sample,
                         "label": detection["label"], "boxes": []}
                tracks.append(track)
            else:
                available.remove(track)
            track["last"] = sample
            track["last_box"] = detection["bbox"]
            track["boxes"].append(detection["bbox"])
            next_active.append(track)
        active = next_active
    regions = []
    for track in tracks:
        first, last = track["first"], track["last"]
        start = 0 if first == 0 else (
            observations[first-1]["frame"] + observations[first]["frame"] + 1) // 2
        end = total_frames if last == len(observations)-1 else (
            observations[last]["frame"] + observations[last+1]["frame"] + 1) // 2
        boxes = track["boxes"]
        box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
               max(b[2] for b in boxes), max(b[3] for b in boxes)]
        # Vision models sometimes locate only lettering and clip an adjoining
        # emblem. Visible suggestion margins include that fringe for review.
        horizontal, vertical = (box[2]-box[0])*.25, (box[3]-box[1])*.15
        box = [max(0, box[0]-horizontal), max(0, box[1]-vertical),
               min(1, box[2]+horizontal), min(1, box[3]+vertical)]
        regions.append({"id": str(uuid.uuid4()), "bbox": box,
                        "start_frame": start, "end_frame": end,
                        "label": track["label"], "source": "suggested",
                        "_first_sample": first, "_last_sample": last})
    return regions


def refine_ranges(regions, observations, observe):
    """Ask vision about intermediate frames to narrow each observed transition.

    Binary search assumes one presence transition between neighboring samples.
    It cannot discover watermarks appearing entirely between coarse samples.
    """
    for region in regions:
        first, last = region.pop("_first_sample"), region.pop("_last_sample")
        def present(frame):
            return any(box_iou(region["bbox"], item["bbox"]) >= .2
                       for item in observe(frame))
        if first > 0:
            low, high = observations[first-1]["frame"], observations[first]["frame"]
            step = max(1, high-low)
            while low > 0 and present(low):
                high, low = low, max(0, low-step)
            if low == 0 and present(0):
                region["start_frame"] = 0
                high = 0
            while high-low > 1:
                mid = (low+high)//2
                if present(mid):
                    high = mid
                else:
                    low = mid
            region["start_frame"] = high
        if last < len(observations)-1:
            low, high = observations[last]["frame"], observations[last+1]["frame"]
            final_frame = observations[-1]["frame"]
            step = max(1, high-low)
            while high < final_frame and present(high):
                low, high = high, min(final_frame, high+step)
            if high == final_frame and present(high):
                high = final_frame+1
            while high-low > 1:
                mid = (low+high)//2
                if present(mid):
                    low = mid
                else:
                    high = mid
            region["end_frame"] = high
    return regions


def verify_range_boundaries(regions, total_frames, observe):
    """Independently check the first/last included and neighboring excluded frames."""
    checks = 0
    for region in regions:
        start, end = region["start_frame"], region["end_frame"]
        def present(frame):
            return any(box_iou(region["bbox"], item["bbox"]) >= .2
                       for item in observe(frame))
        for frame, expected in [(start, True), (end - 1, True),
                                (start - 1, False), (end, False)]:
            if not 0 <= frame < total_frames:
                continue
            checks += 1
            if present(frame) != expected:
                raise ValueError("A watermark boundary could not be confirmed. "
                                 "The source has not been cleaned. Rescan at a shorter interval.")
        region["timing_verified"] = True
    return checks


def recurring_cadence(regions, fps):
    """Describe an observed repeating overlay, without inventing unseen intervals."""
    groups = []
    for region in sorted(regions, key=lambda r: r["start_frame"]):
        group = next((g for g in groups if box_iou(g[0]["bbox"], region["bbox"]) >= .35), None)
        if group is None:
            groups.append([region])
        else:
            group.append(region)
    group = max(groups, key=len, default=[])
    if len(group) < 4:
        return {}
    differences = [(b["start_frame"] - a["start_frame"]) / fps
                   for a, b in zip(group, group[1:])]
    period = statistics.median(differences)
    deviation = statistics.median(abs(d - period) for d in differences)
    if period <= 0 or deviation > period * .2:
        return {}
    return {"period_seconds": round(period, 3), "appearances": len(group),
            "median_deviation_seconds": round(deviation, 3)}


def local_endpoint(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Use a local HTTP vision server. Video frames stay on this computer.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Invalid local server URL.")
    return url.rstrip("/")
