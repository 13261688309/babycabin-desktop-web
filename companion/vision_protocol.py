"""64-byte observation-only USB protocol; CRC is integrity, not authentication."""

import math
import re
import struct
import zlib

MODEL_SHA256 = '2adf7e59db4a3263b209fb43f955192cd2fddae08539af04c179c9f4e44085e7'
TTL_MS = 1500
LABEL_IDS = {'Unknown': 0, 'Cry': 1, 'Happy': 2, 'Neutral': 3, 'Sleep': 4}
FRAME = struct.Struct('<4sIIIIHBB32sII')
LINK_FIELDS = frozenset(('v', 'ready', 'model', 'host', 'seq', 'label', 'conf', 'age_ms',
                         'fresh', 'accepted', 'rejected', 'drops', 'alerts', 'now_ms'))


def parse_vision_link(line):
    text = line.decode('ascii')
    if not text.startswith('vision_link ') or len(text) > 512:
        raise ValueError('Invalid vision capability line')
    data = {}
    for token in text[len('vision_link '):].split(' '):
        key, separator, value = token.partition('=')
        if not separator or key in data:
            raise ValueError('Missing or duplicate vision field')
        if key == 'model':
            if not re.fullmatch('[0-9a-f]{64}', value):
                raise ValueError('Invalid model digest')
            data[key] = value
        else:
            if not re.fullmatch('[0-9]{1,10}', value) or int(value) > 0xFFFFFFFF:
                raise ValueError('Invalid uint32 vision field')
            data[key] = int(value)
    if (set(data) != LINK_FIELDS or data['v'] != 1 or data['ready'] != 1 or
            data['label'] not in LABEL_IDS.values() or data['conf'] > 1000 or data['fresh'] not in (0, 1)):
        raise ValueError('Unsupported vision capability schema')
    if not data['fresh'] and (data['label'] != 0 or data['conf'] != 0 or data['age_ms'] != 0xFFFFFFFF):
        raise ValueError('Stale vision must be unknown')
    if data['fresh'] and (data['age_ms'] >= TTL_MS or
            (data['label'] == 0 and data['conf'] != 0) or (data['label'] != 0 and data['conf'] < 600)):
        raise ValueError('Invalid fresh vision bounds')
    return data


def encode_observation(*, host, sequence, advertised_ms, label, confidence, frame_age_ms):
    values = (host, sequence, advertised_ms, frame_age_ms)
    if (any(type(v) is not int for v in values) or not 1 <= host <= 0xFFFFFFFF or
            not 0 <= sequence <= 0xFFFFFFFF or not 0 <= advertised_ms <= 0xFFFFFFFF or
            not 0 <= frame_age_ms < TTL_MS or label not in LABEL_IDS or
            type(confidence) is not int or not 0 <= confidence <= 1000 or
            (label == 'Unknown' and confidence != 0) or (label != 'Unknown' and confidence < 600)):
        raise ValueError('Invalid observation packet fields')
    expiry = (advertised_ms + TTL_MS - frame_age_ms) & 0xFFFFFFFF
    packet = FRAME.pack(b'BV01', host, sequence, expiry, frame_age_ms, confidence,
                        LABEL_IDS[label], int(label != 'Unknown'), bytes.fromhex(MODEL_SHA256), 1, 0)
    return packet[:-4] + struct.pack('<I', zlib.crc32(packet[:-4]))


def observation_fresh(item):
    data = item.get('data') or {}
    age, used = item.get('ageMs'), data.get('age_ms')
    return (item.get('fresh') is True and data.get('fresh') == 1 and
            type(age) is int and type(used) is int and 0 <= age and 0 <= used and age + used < TTL_MS)


def confirmed_sample(record, now_ms):
    observation = record.get('observation') or {}
    if observation.get('reason') != 'confirmed' or observation.get('state') not in LABEL_IDS or observation['state'] == 'Unknown':
        return None
    captured = record.get('frameCapturedMs')
    confidence = observation.get('confidence')
    if (record.get('sourceKind') != 'network' or (record.get('model') or {}).get('sha256') != MODEL_SHA256 or
            (record.get('video') or {}).get('connectionState') != 'streaming' or
            type(captured) is not int or not 0 <= now_ms - captured < TTL_MS or
            type(confidence) not in (int, float) or not math.isfinite(confidence) or not .6 <= confidence <= 1):
        return None
    return dict(label=observation['state'], confidence=int(confidence * 1000),
                frame_age_ms=now_ms - captured, captured_ms=captured)
