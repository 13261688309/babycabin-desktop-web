"""Sample the local GB01 console without sending actuator or camera commands."""
import argparse
from collections import Counter
import json
import math
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

URL = 'http://127.0.0.1:8774/api/state'
PERIOD = 0.1
FRESHNESS_MS = 400


def p95(values):
    return sorted(values)[math.ceil(len(values) * .95) - 1] if values else None


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_state(opener):
    with opener.open(URL, timeout=1) as response:
        return json.loads(response.read(256 * 1024 + 1))


def check_idle(state):
    if (state.get('profile') != 'GB01-idle-low' or state.get('mode') != 'idle'
            or state.get('tiltReady') is not False):
        raise ValueError('Expected idle GB01 with wiring enable unchecked.')
    if state.get('controlState') == 'offline' and state.get('status') is None:
        return
    status = state.get('status') or {}
    if (state.get('controlState') != 'idle'
            or any(type(status.get(key)) is not int or status[key] != 0
                   for key in ('pwm', 'lease', 'fault', 'motor', 'rock'))):
        raise ValueError('Expected a disconnected controller or idle controller feedback without faults.')


def summarize(samples):
    ages = []
    first_age = {}
    fresh_faces = {}
    counters = []
    transport_states = Counter()
    transitions = []
    latest_transport_state = None
    stale = missing = 0
    for state in samples:
        frame_id, age = state.get('frameId'), state.get('frameAgeMs')
        camera = state['camera']
        count = camera.get('frames')
        transport = camera.get('transport')
        transport_state = (transport if transport in ('mjpeg', 'snapshot') else 'none', camera['state'])
        transport_states['/'.join(transport_state)] += 1
        if transport_state != latest_transport_state:
            transitions.append(dict(atMs=state.get('sampleMs'), transport=transport_state[0],
                                    state=transport_state[1], reason=camera.get('fallbackReason')))
            latest_transport_state = transport_state
        if type(count) is int and count >= 0:
            counters.append(count)
        if (state['camera']['state'] != 'streaming' or type(frame_id) is not int
                or type(age) is not int or age < 0):
            missing += 1
            continue
        ages.append(age)
        first_age.setdefault(frame_id, age)
        fresh = state.get('frameFresh') is True and age < FRESHNESS_MS
        stale += not fresh
        if fresh and type(state.get('faceCount')) is int:
            fresh_faces.setdefault(frame_id, state['faceCount'])
    ids = sorted(first_age)
    gaps = [b - a for a, b in zip(ids, ids[1:])]
    capture_increase = (counters[-1] - counters[0]
                        if len(counters) >= 2 and counters == sorted(counters) else None)
    faces = Counter(min(count, 2) for count in fresh_faces.values())
    return dict(samples=len(samples), observedDistinctFrames=len(ids),
                missingSamples=missing, staleSamples=stale,
                p95AgeMs=p95(ages), maxAgeMs=max(ages, default=None),
                freshnessLimitMs=FRESHNESS_MS,
                capturedFrameCounterIncrease=capture_increase,
                observedAnalysisGapP95Ms=p95(gaps), observedAnalysisGapMaxMs=max(gaps, default=None),
                gapsAtLeastFreshnessLimit=sum(gap >= FRESHNESS_MS for gap in gaps),
                firstObservedAnalysisFrameAgeP95Ms=p95(list(first_age.values())),
                freshFaceFrames=dict(noFace=faces[0], oneFace=faces[1], multipleFaces=faces[2]),
                transportStateSamples=dict(transport_states), transportTransitions=transitions)


def observe(seconds, get_state, *, clock=time.monotonic, sleep=time.sleep):
    start = clock()
    samples = []
    camera_generation = None
    while clock() - start < seconds:
        tick = clock()
        state = get_state()
        check_idle(state)
        camera = state['camera']
        generation = camera['generation']
        if not samples:
            if camera['state'] != 'streaming' or type(generation) is not int:
                raise ValueError('Connect the camera on the existing console before sampling.')
            camera_generation = generation
        if generation != camera_generation:
            raise ValueError('Camera generation changed during sampling; results were not combined.')
        # Retain timing and transport metadata, not tokens, addresses or face boxes.
        faces = state.get('faces')
        reason = camera.get('fallbackReason')
        samples.append(dict(sampleMs=round((tick-start)*1000),
                            camera=dict(state=camera['state'], frames=camera.get('frames'),
                                        transport=camera.get('transport'),
                                        fallbackReason=reason if isinstance(reason,str) and reason.isidentifier() else None),
                            frameId=state.get('frameId'), frameAgeMs=state.get('frameAgeMs'),
                            frameFresh=state.get('frameFresh'),
                            faceCount=len(faces) if isinstance(faces, list) else None))
        sleep(max(0, min(tick + PERIOD, start + seconds) - clock()))
    return dict(elapsedSeconds=round(clock() - start, 2), **summarize(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, choices=range(1, 121), metavar='1..120', default=20)
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        result = observe(args.seconds, lambda: read_state(opener))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(2, f'Observation stopped: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
