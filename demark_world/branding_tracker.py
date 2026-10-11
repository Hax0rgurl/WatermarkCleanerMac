"""Track the appearance of a brand already confirmed by vision and exact-name OCR."""

import cv2
import numpy as np

from demark_world.vision_scan import box_iou


class BrandingTracker:
    def __init__(self):
        self.references = []

    def add(self, frame, anchor, box):
        # Several backgrounds help with translucent branding. Never use generic
        # white text as a reference: callers supply model-confirmed name anchors.
        if sum(box_iou(r["bbox"], box) >= .35 for r in self.references) >= 3:
            return
        h, w = frame.shape[:2]
        a = anchor["bbox"]
        x1, y1 = max(0, round(a[0]*w)-2), max(0, round(a[1]*h)-2)
        x2, y2 = min(w, round(a[2]*w)+2), min(h, round(a[3]*h)+2)
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if min(gray.shape) < 5 or gray.size < 100 or gray.std() < 4:
            return
        edges = cv2.Canny(gray, 50, 120)
        if np.count_nonzero(edges) < 20:
            return
        self.references.append({"rect": (x1,y1,x2,y2), "gray": gray, "edges": edges,
                                "bbox": list(box), "label": anchor["text"]})

    def detect(self, frame):
        h, w = frame.shape[:2]
        found = []
        for ref in self.references:
            if any(box_iou(ref["bbox"], d["bbox"]) >= .35 for d in found):
                continue
            x1,y1,x2,y2 = ref["rect"]
            slack = max(5, round((y2-y1)*.2))
            crop = frame[max(0,y1-slack):min(h,y2+slack),
                         max(0,x1-slack):min(w,x2+slack)]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] < ref["gray"].shape[0] or gray.shape[1] < ref["gray"].shape[1]:
                continue
            # Require both intensity structure and the same lettering edge pattern
            # at the same offset. Bright captions or a textured background alone
            # cannot satisfy a match to the confirmed brand's glyphs.
            intensity = cv2.matchTemplate(gray, ref["gray"], cv2.TM_CCOEFF_NORMED)
            edges = cv2.matchTemplate(cv2.Canny(gray,50,120), ref["edges"], cv2.TM_CCOEFF_NORMED)
            if np.any((intensity >= .65) & (edges >= .4)):
                found.append({"bbox": ref["bbox"], "label": ref["label"],
                              "method": "confirmed-brand-pattern"})
        return found
