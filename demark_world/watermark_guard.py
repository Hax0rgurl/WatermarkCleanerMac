"""Independent, conservative identity checks for automatic suggestions."""

import base64
import json
import os
from pathlib import Path
import re
import select
import subprocess
import unicodedata

from demark_world.configs import SOURCE_ROOT


def _words(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def target_names(value: str) -> list[str]:
    names = []
    # People copy the complete visible branding, including an icon description
    # or a creator suffix. Recognizable brand names remain the identity anchors.
    brands = ["invideo", "sora", "storyblocks", "istock", "getty images"]
    for part in value.split(","):
        if not part.strip():
            continue
        cleaned = re.sub(r":[a-zA-Z0-9_+\-]+:", " ", part)
        recognized = [brand for brand in brands if matches_target(cleaned, [brand])]
        names.extend(recognized or [" ".join(_words(cleaned))])
    names = list(dict.fromkeys(names))
    if not names or any(len(name.replace(" ", "")) < 3 for name in names):
        raise ValueError("Enter the specific watermark name, such as InVideo or Sora. "
                         "For a symbol without readable text, draw a timed box.")
    if len(names) > 8 or any(len(name) > 100 for name in names):
        raise ValueError("Enter up to eight specific watermark names.")
    return names


def matches_target(text, names):
    words = _words(text)
    # Match complete words or complete consecutive word groups. A name such as
    # Sora must not match unrelated words such as Soraya.
    return any("".join(words[start:stop]) == name.replace(" ", "")
               for name in names for start in range(len(words))
               for stop in range(start+1, min(len(words), start+8)+1))


def text_anchors(texts, names):
    return [text for text in texts if text.get("confidence", 0) >= .5
            and matches_target(text.get("text", ""), names)]


def verified_detections(detections, texts, names):
    """A model label is insufficient: matching text must be inside its box."""
    anchors = text_anchors(texts, names)
    accepted = []
    for detection in detections:
        box = detection["bbox"]
        width, height = box[2]-box[0], box[3]-box[1]
        if width*height > .12:
            continue
        for anchor in anchors:
            text_box = anchor["bbox"]
            tw, th = text_box[2]-text_box[0], text_box[3]-text_box[1]
            intersection = max(0, min(box[2],text_box[2])-max(box[0],text_box[0])) * max(
                0, min(box[3],text_box[3])-max(box[1],text_box[1]))
            if tw*th > 0 and intersection/(tw*th) >= .7 and (
                width <= max(.35,tw*3) and height <= max(.1,th*3)
            ):
                accepted.append(detection)
                break
    return accepted


def named_text_detections(texts, names, width, height):
    detections = []
    for anchor in text_anchors(texts, names):
        x1,y1,x2,y2 = anchor["bbox"]
        text_height = y2-y1
        # Include an adjoining logo icon and the taller graphic beside the name.
        box = [max(0,x1-text_height*height/width*1.4), max(0,y1-text_height*.45),
               min(1,x2), min(1,y2+text_height*.45)]
        detections.append({"label":anchor["text"], "bbox":box})
    return verified_detections(detections, texts, names)


def anchor_model_detections(detections, texts, names, width, height):
    """Complete an undersized model box using the independently read same name.

    The model must name the target and overlap its observed lettering. This
    cannot expand a box onto unrelated words or onto branding elsewhere.
    """
    completed = []
    for detection in detections:
        if not matches_target(detection.get("label", ""), names):
            continue
        box = detection["bbox"]
        for named in named_text_detections(texts, names, width, height):
            other = named["bbox"]
            overlap = max(0, min(box[2], other[2])-max(box[0], other[0])) * max(
                0, min(box[3], other[3])-max(box[1], other[1]))
            area = (box[2]-box[0])*(box[3]-box[1])
            if area <= 0 or overlap/area < .5:
                continue
            combined = [min(box[0], other[0]), min(box[1], other[1]),
                        max(box[2], other[2]), max(box[3], other[3])]
            completed.append({**detection, "bbox": combined})
    return verified_detections(completed, texts, names)


class LocalTextReader:
    def __init__(self, helper=None):
        candidates = [Path(helper or os.environ.get("MAC_WATERMARK_OCR_HELPER", "")),
                      SOURCE_ROOT.parent/"tools/vision-ocr",
                      SOURCE_ROOT/".build/macos/vision-ocr"]
        executable = next((path for path in candidates if path.is_file()), None)
        if executable is None:
            raise RuntimeError("The local watermark text check is unavailable. "
                               "Use manual timed boxes.")
        self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, bufsize=1)

    def read(self, frame):
        import cv2
        ok, image = cv2.imencode(".png", frame)
        if not ok:
            raise ValueError("Could not prepare the independent text check.")
        self.process.stdin.write(json.dumps({"image":base64.b64encode(image).decode()})+"\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 45)[0]:
            raise RuntimeError("The local watermark text check timed out. No boxes were accepted.")
        answer = json.loads(self.process.stdout.readline())
        if "error" in answer:
            raise RuntimeError("The local watermark text check failed: "+answer["error"])
        if not isinstance(answer.get("text"), list):
            raise ValueError("Invalid local text-check result.")
        return answer["text"]

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
