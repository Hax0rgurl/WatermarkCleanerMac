import cv2
import numpy as np

from demark_world.branding_tracker import BrandingTracker


def test_tracking_matches_confirmed_brand_and_rejects_other_white_words():
    ref = np.full((180, 640, 3), 35, np.uint8)
    cv2.putText(ref, "InVideo", (460,45), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    anchor = {"text": "InVideo", "bbox": [455/640,25/180,545/640,50/180]}
    box = [450/640,20/180,550/640,55/180]
    tracker = BrandingTracker()
    tracker.add(ref, anchor, box)
    changed_background = np.full_like(ref, 95)
    cv2.putText(changed_background, "InVideo", (460,45), cv2.FONT_HERSHEY_SIMPLEX, .7, (225,225,225), 2)
    assert tracker.detect(changed_background)
    caption = np.full_like(ref, 35)
    cv2.putText(caption, "FOCUS", (460,45), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    assert tracker.detect(caption) == []
    assert tracker.detect(np.full_like(ref, 255)) == []
