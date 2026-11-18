"""Score two face detectors on the same local preview JPEGs without saving frames."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from urllib.request import ProxyHandler, build_opener

from companion.face_yunet import MODEL_SHA256, YuNetFaceDetector
from companion.gimbal import FaceDetector, FaceTracker, valid_face
from tools.check_gimbal_frame_age import FRESHNESS_MS, NoRedirect, URL, check_idle, read_state


MAX_JPEG = 512 * 1024
MIN_PAIRS = 12
SCENES = ('front', 'slight-side', 'empty')


@dataclass(frozen=True)
class Preview:
    generation: int
    frame_id: int
    age_ms: int
    jpeg: bytes


def read_preview(opener, after):
    with opener.open(URL.removesuffix('/api/state')+f'/video.jpg?after={after}', timeout=1) as response:
        if response.status == 204:
            return None
        if response.status != 200 or response.headers.get_content_type() != 'image/jpeg':
            raise ValueError('Expected a local preview JPEG')
        frame = Preview(int(response.headers['X-Camera-Generation']),
                        int(response.headers['X-Frame-Id']),
                        int(response.headers['X-Frame-Age-Ms']),
                        response.read(MAX_JPEG+1))
        if not 0 < len(frame.jpeg) <= MAX_JPEG:
            raise ValueError('Invalid preview JPEG length')
        return frame


def paired_observation(seconds, scene, get_state, get_preview, decode, detectors,
                       *, clock=time.monotonic, sleep=time.sleep):
    if scene not in SCENES:
        raise ValueError('Choose a known scene')
    started = clock()
    before = get_state()
    check_idle(before)
    camera = before['camera']
    generation = camera['generation']
    if camera['state'] != 'streaming' or type(generation) is not int:
        raise ValueError('The original page must have a live preview before this phase')

    counts = {name: Counter() for name in detectors}
    timings = {name: [] for name in detectors}
    trackers = {name: FaceTracker() for name in detectors}
    preview = Counter()
    after = 0
    while clock()-started < seconds:
        state = get_state()
        check_idle(state)
        if state['camera']['generation'] != generation:
            raise ValueError('Camera generation changed; discarded this phase')
        if state['camera']['state'] != 'streaming':
            preview['streamUnavailablePolls'] += 1
            for tracker in trackers.values():
                tracker.reset()
            sleep(.05)
            continue
        requested = clock()
        sample = get_preview(after)
        if sample is None:
            preview['noNewFramePolls'] += 1
            sleep(.01)
            continue
        if sample.generation != generation:
            raise ValueError('Preview generation changed; discarded this phase')
        if sample.frame_id <= after or sample.age_ms < 0:
            raise ValueError('Preview frame was repeated or its age was invalid')
        after = sample.frame_id
        if clock()-started >= seconds:
            break
        frame = decode(sample.jpeg)
        # Count local transfer and decode against both models, not the other model's compute time.
        age = sample.age_ms + math.ceil((clock()-requested)*1000)
        if age >= FRESHNESS_MS:
            preview['staleFrames'] += 1
            continue
        if frame is None:
            preview['decodeFailures'] += 1
            continue
        height, width = frame.shape[:2]
        preview['pairedFreshFrames'] += 1
        for name, detector in detectors.items():
            began = clock()
            boxes = detector.detect(frame)
            elapsed = (clock()-began)*1000
            timings[name].append(elapsed)
            if any(not valid_face(box, width, height) for box in boxes):
                raise ValueError(f'{name} returned an invalid box; discarded this phase')
            row = counts[name]
            row['anyBoxFrames'] += bool(boxes)
            row['exactlyOneBoxFrames'] += len(boxes) == 1
            row['multipleBoxFrames'] += len(boxes) > 1
            finished_age = age + math.ceil(elapsed)
            row['pastDeadlineFrames'] += finished_age >= FRESHNESS_MS
            row['oneBoxWithinDeadlineFrames'] += len(boxes) == 1 and finished_age < FRESHNESS_MS
            tracker = trackers[name]
            finished_at = sample.frame_id+finished_age
            if tracker.last_capture is not None and finished_at-tracker.last_capture >= FRESHNESS_MS:
                tracker.reset()
            target = tracker.update(boxes, width, height, sample.frame_id, finished_at)
            row['simulatedLockFrames'] += target is not None
    result = {}
    pairs = preview['pairedFreshFrames']
    for name, row in counts.items():
        ms = sorted(timings[name])
        result[name] = dict(row, detectorP95Ms=round(ms[math.ceil(.95*len(ms))-1], 2) if ms else None)
        if pairs:
            rate = 'unexpectedBoxRate' if scene == 'empty' else 'exactlyOneBoxWithinDeadlineRate'
            numerator = row['anyBoxFrames' if scene == 'empty' else 'oneBoxWithinDeadlineFrames']
            result[name][rate] = round(numerator/pairs, 4)
    return dict(scene=scene, sceneLabel='operator-declared', seconds=seconds,
                cameraGeneration=generation, pairedFreshFrames=pairs, enoughPairs=pairs>=MIN_PAIRS,
                missing=dict(staleFrames=preview['staleFrames'], noNewFramePolls=preview['noNewFramePolls'],
                             streamUnavailablePolls=preview['streamUnavailablePolls'],
                             decodeFailures=preview['decodeFailures']),
                yunetModelSha256=MODEL_SHA256, results=result)


def main():
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SCENES, required=True)
    parser.add_argument('--seconds', type=int, choices=range(4, 21), metavar='4..20', default=8)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        detectors = dict(haar=FaceDetector(), yunet=YuNetFaceDetector(args.weights))
        warmup = np.zeros((240, 320, 3), dtype=np.uint8)
        for detector in detectors.values():
            detector.detect(warmup)
        result = paired_observation(args.seconds, args.scene, lambda: read_state(opener),
            lambda after: read_preview(opener, after),
            lambda data: cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR), detectors)
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        parser.exit(2, f'Observation stopped: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
