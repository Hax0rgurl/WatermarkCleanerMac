"""Isolated optional MLX process. JSON lines are the only stdout protocol."""

import argparse
import base64
import contextlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from demark_world.vision_scan import (
    VISION_MODELS, WATERMARK_SCHEMA, PROMPT, local_endpoint, parse_watermarks,
    box_iou, ranges_from_observations, refine_ranges, sample_indices,
    verify_range_boundaries, recurring_cadence,
)
from demark_world.watermark_guard import (
    LocalTextReader, named_text_detections, target_names, text_anchors, anchor_model_detections,
)


PROTOCOL_STDOUT = sys.stdout
EVENT_LOCK = threading.Lock()


def emit(**event):
    with EVENT_LOCK:
        print(json.dumps(event), file=PROTOCOL_STDOUT, flush=True)


def run(request):
    import cv2
    from demark_world.branding_tracker import BrandingTracker
    names = target_names(request.get("watermark_text", ""))
    text_reader = None
    capture = cv2.VideoCapture(request["path"])
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the selected video for scanning.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_indices(count, fps, float(request.get("sampling_seconds", 1)))
        profile = request.get("profile", "Quick")
        model_id, revision = VISION_MODELS[profile]
        text_reader = LocalTextReader()
        search_prompt = PROMPT + "\nFind ONLY this specific watermark identity: " + ", ".join(names) + (
            ". All other text must be left alone. The target name inside a subtitle, "
            "title, chapter heading or an object in the scene is still NOT a watermark.")
        endpoint = request.get("endpoint")
        use_model = request.get("use_vision_model", True)
        if not use_model:
            model_id = "Apple Vision (named text)"
            emit(status="Scanning only the named watermark with local text recognition…")
        elif endpoint:
            import requests
            endpoint = local_endpoint(endpoint)
            model_id = request.get("server_model", "")
            if not model_id:
                raise ValueError("Enter the model name used by your local vision server.")
        else:
            # Use ordinary HTTP transfers, which also work on networks where the
            # optional Xet CDN cannot resolve. No change to other apps' download settings.
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            emit(status=f"Loading {model_id}. First use downloads its weights.")
            with contextlib.redirect_stdout(sys.stderr):
                from huggingface_hub import snapshot_download
                from mlx_vlm import load, generate
                from mlx_vlm.prompt_utils import apply_chat_template
                from mlx_vlm.utils import load_config
                from mlx_vlm.structured import build_json_schema_logits_processor
                import mlx.core as mx
                mx.set_cache_limit(64*1024**2)
                from tqdm import tqdm
                class DownloadProgress(tqdm):
                    def display(self, *args, **kwargs):
                        now = time.monotonic()
                        if getattr(self, "unit", "") == "B" and self.n and (
                            now-getattr(self, "last_report", 0) > 1):
                            self.last_report = now
                            total = self.total or 0
                            emit(progress=min(1, self.n/total) if total else 0,
                                 status=f"Downloading vision weights: {self.n/1024**3:.2f} GB "
                                        f"of {total/1024**3:.2f} GB")
                # Reuse a complete compatible cached snapshot before downloading.
                cached = request.get("cached_model")
                model_path = cached or snapshot_download(model_id, revision=revision,
                                                         tqdm_class=DownloadProgress)
                model, processor = load(model_path)
                config = load_config(model_path)
        answers, text_answers, text_frames, semantic_observations = {}, {}, {}, []
        locations, negative_versions = [], {}
        def read_frame(frame_index):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {frame_index}. Scan was not completed.")
            return frame

        def observe_text(frame_index):
            if frame_index in text_answers and (text_answers[frame_index] or
                    negative_versions.get(frame_index) == len(locations)):
                return text_answers[frame_index]
            frame = read_frame(frame_index)
            texts = list(text_frames.get(frame_index, [])) if frame_index in text_frames else text_reader.read(frame)
            h, w = frame.shape[:2]
            if not text_anchors(texts, names):
                # After discovering the brand, magnify its actual screen locations.
                # Small lettering on busy backgrounds/fades can be missed by full-
                # frame OCR. This still requires the same name to be read locally.
                for box in locations:
                    padding = max(12, (box[3]-box[1])*h*.5)
                    left, top = max(0,int(box[0]*w-padding)), max(0,int(box[1]*h-padding))
                    right, bottom = min(w,int(box[2]*w+padding)), min(h,int(box[3]*h+padding))
                    crop = frame[top:bottom,left:right]
                    scale = min(4, 1200/max(1,right-left))
                    enlarged = cv2.resize(crop, (round((right-left)*scale),round((bottom-top)*scale)))
                    for text in text_reader.read(enlarged):
                        if not text_anchors([text], names):
                            continue
                        b = text["bbox"]
                        texts.append({**text, "bbox": [(left+b[0]*(right-left))/w,
                                                       (top+b[1]*(bottom-top))/h,
                                                       (left+b[2]*(right-left))/w,
                                                       (top+b[3]*(bottom-top))/h]})
            text_frames[frame_index] = texts
            text_answers[frame_index] = named_text_detections(texts, names, w, h)
            for detection in text_answers[frame_index]:
                if len(locations) < 8 and not any(box_iou(detection["bbox"], b) >= .35 for b in locations):
                    locations.append(detection["bbox"])
            if not text_answers[frame_index]:
                negative_versions[frame_index] = len(locations)
            return text_answers[frame_index]

        def observe(frame_index):
            if frame_index in answers:
                return answers[frame_index]
            observe_text(frame_index)
            texts = text_frames[frame_index]
            anchors = text_anchors(texts, names)
            if not anchors:
                answers[frame_index] = []
                return []
            if not use_model:
                answers[frame_index] = text_answers[frame_index]
                return answers[frame_index]
            frame = read_frame(frame_index)
            # Preserve logo detail while bounding vision memory on large videos.
            h, w = frame.shape[:2]
            if max(h, w) > 1280:
                frame = cv2.resize(frame, (round(w*1280/max(h,w)), round(h*1280/max(h,w))))
            # Inspect a bounded context around independently recognized branding,
            # instead of allocating vision tokens for every word in the full frame.
            detections = []
            h, w = frame.shape[:2]
            for anchor in anchors:
                b = anchor["bbox"]
                cw, ch = max(384, (b[2]-b[0])*w*3), max(160, (b[3]-b[1])*h*4)
                cx, cy = (b[0]+b[2])*w/2, (b[1]+b[3])*h/2
                left, top = max(0,int(cx-cw/2)), max(0,int(cy-ch/2))
                right, bottom = min(w,int(cx+cw/2)), min(h,int(cy+ch/2))
                crop = frame[top:bottom,left:right]
                if endpoint:
                    ok, jpg = cv2.imencode(".jpg", crop)
                    if not ok:
                        raise ValueError("Could not prepare a frame for the local vision server.")
                    data = base64.b64encode(jpg).decode()
                    response = requests.post(endpoint + "/chat/completions", json={
                        "model": model_id, "temperature": 0, "max_tokens": 768,
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": search_prompt},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"+data}}
                        ]}], "chat_template_kwargs": {"enable_thinking": False},
                    }, timeout=(10, 180), allow_redirects=False)
                    if response.is_redirect:
                        raise ValueError("The local vision server redirected the request. Frames were not forwarded.")
                    response.raise_for_status()
                    answer = response.json()["choices"][0]["message"]["content"]
                else:
                    from PIL import Image
                    picture = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    with contextlib.redirect_stdout(sys.stderr):
                        prompt = apply_chat_template(processor, config, search_prompt,
                                                     num_images=1, enable_thinking=False)
                        grammar = build_json_schema_logits_processor(processor.tokenizer,
                                                                     WATERMARK_SCHEMA)
                        answer = generate(model, processor, prompt, image=[picture],
                                          logits_processors=[grammar], max_tokens=1024,
                                          temperature=0, verbose=False).text
                        mx.clear_cache()
                parsed = parse_watermarks(answer)
                semantic_observations.append({"frame": frame_index, "anchor": anchor["text"],
                                              "crop_bbox": [left/w,top/h,right/w,bottom/h],
                                              "response": parsed})
                for detection in parsed:
                    x1,y1,x2,y2 = detection["bbox"]
                    detection["bbox"] = [(left+x1*(right-left))/w,(top+y1*(bottom-top))/h,
                                         (left+x2*(right-left))/w,(top+y2*(bottom-top))/h]
                    if not any(box_iou(detection["bbox"],d["bbox"]) >= .7 for d in detections):
                        detections.append(detection)
            answers[frame_index] = anchor_model_detections(detections, texts, names, w, h)
            return answers[frame_index]
        observations = []
        refine = request.get("refine", True)
        for position, frame_index in enumerate(indices):
            observations.append({"frame": frame_index, "watermarks": observe_text(frame_index)})
            emit(progress=(position+1)/len(indices)*.55,
                 status=f"Finding {request['watermark_text']} appearances ({position+1}/{len(indices)})…")
        # Newly discovered locations also get checked in earlier negative samples.
        for observation in observations:
            if not observation["watermarks"]:
                observation["watermarks"] = observe_text(observation["frame"])
        regions = ranges_from_observations(observations, count)
        tracker = BrandingTracker()
        if use_model:
            # Confirm the identity once per appearance, then use the independently
            # read branding to locate its boundaries. Large vision inference on
            # every fade frame is slower and less consistent than this identity gate.
            confirmed = []
            for position, region in enumerate(regions):
                first, last = region["_first_sample"], region["_last_sample"]
                candidates = list(dict.fromkeys([(first + last)//2, first, last]))
                emit(progress=.55 + (position/max(1,len(regions)))*.25,
                     status=f"Vision model checking branding {position+1}/{len(regions)}…")
                for sample in candidates:
                    detections = observe(observations[sample]["frame"])
                    matching = [d for d in detections if box_iou(region["bbox"], d["bbox"]) >= .2]
                    if matching:
                        boxes = [region["bbox"]] + [d["bbox"] for d in matching]
                        region["bbox"] = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                          max(b[2] for b in boxes), max(b[3] for b in boxes)]
                        confirmed.append(region)
                        for anchor in text_anchors(text_frames[observations[sample]["frame"]], names):
                            if box_iou(region["bbox"], anchor["bbox"]) >= .1:
                                tracker.add(read_frame(observations[sample]["frame"]), anchor, region["bbox"])
                        break
            regions = confirmed
        timing_answers = {}
        def observe_timing(index):
            if index not in timing_answers:
                detections = list(observe_text(index))
                if tracker.references:
                    for detection in tracker.detect(read_frame(index)):
                        if not any(box_iou(detection["bbox"], d["bbox"]) >= .35 for d in detections):
                            detections.append(detection)
                timing_answers[index] = detections
            return timing_answers[index]
        if refine:
            def boundary_frame(index):
                emit(progress=.85, status=f"Checking a start/stop at {index/fps:.2f} seconds…")
                return observe_timing(index)
            regions = refine_ranges(regions, observations, boundary_frame)
            emit(progress=.96, status="Verifying first, last and neighboring frames for each appearance…")
            boundary_checks = verify_range_boundaries(regions, count, observe_timing)
        else:
            boundary_checks = 0
            for region in regions:
                region.pop("_first_sample", None)
                region.pop("_last_sample", None)
        for region in regions:
            region["verification"] = "named-ocr-v1"
        emit(result={"regions": regions,
                     "observations": observations, "fps": fps, "total_frames": count,
                     "sampling_seconds": request.get("sampling_seconds", 1),
                     "refined": refine, "model": model_id,
                     "timing_verified": refine, "boundary_checks": boundary_checks,
                     "timing_method": "confirmed brand tracking + OCR" if tracker.references else "exact-name OCR",
                     "tracking_references": len(tracker.references),
                     "cadence": recurring_cadence(regions, fps),
                     "watermark_text": request["watermark_text"], "independent_text_check": True,
                     "semantic_model_check": use_model,
                     "semantic_observations": semantic_observations})
    finally:
        capture.release()
        if text_reader is not None:
            text_reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args()
    try:
        run(json.loads(Path(args.request).read_text()))
    except Exception as exc:
        message = str(exc)
        if "Insufficient Memory" in message or "out of memory" in message.lower():
            message = ("The vision model ran out of memory while other apps were using it. "
                       "Use the local text scan, or try the Quick model after video generation finishes.")
        emit(error=f"Vision scan failed: {message}")
        raise SystemExit(1)
