"""Opt-in native USB observation uplink. No command or actuator-write API."""

import secrets
import time
from collections import deque

from .native_usb import NativeUsbTelemetry
from .vision_protocol import MODEL_SHA256, TTL_MS, encode_observation, confirmed_sample


class NativeVisionTelemetry(NativeUsbTelemetry):
    def __init__(self, *args, observation_uplink_confirmed=False, **kwargs):
        if observation_uplink_confirmed is not True:
            raise ValueError('Explicit observation uplink confirmation required')
        self._vision_host = secrets.randbelow(0xFFFFFFFF) + 1
        self._vision_sequence = 0
        self._vision_bytes = self._vision_messages = 0
        self._vision_expired_skips = 0
        self._vision_fault = None
        self._vision_last_capture = None
        self._vision_last_label = None
        self._vision_written = deque(maxlen=64)
        super().__init__(*args, **kwargs)

    def publish_vision(self, record):
        now = time.monotonic_ns() // 1000000
        sample = confirmed_sample(record, now)
        observation = record.get('observation') or {}
        captured = record.get('frameCapturedMs')
        new_frame = (record.get('sourceKind') == 'network' and
            (record.get('model') or {}).get('sha256') == MODEL_SHA256 and
            (record.get('video') or {}).get('connectionState') == 'streaming' and
            record.get('inferenceMs') is not None and type(captured) is int and
            0 <= now - captured < TTL_MS and
            (self._vision_last_capture is None or captured > self._vision_last_capture))
        if new_frame:
            if sample is None:
                sample = dict(label='Unknown', confidence=0, frame_age_ms=now - captured, captured_ms=captured)
        else:
            # Waiting for a new inference is not itself loss of the last sample.
            if (observation.get('reason') == 'confirmed' and
                    (record.get('video') or {}).get('connectionState') == 'streaming'):
                return False
            if self._vision_last_label in (None, 'Unknown'):
                return False
            sample = dict(label='Unknown', confidence=0, frame_age_ms=0, captured_ms=None)
        with self._lock:
            capability = self._latest.get('visionLink') or {}
            data = capability.get('data') or {}
            write_now = time.monotonic_ns() // 1000000
            anchor_age = write_now - capability.get('received_ms', -10000)
            if (self._vision_fault or not self._connected or self._stop.is_set() or self._serial is None or
                    not self._thread.is_alive() or
                    data.get('v') != 1 or data.get('ready') != 1 or data.get('model') != MODEL_SHA256 or
                    data.get('host') not in (0, self._vision_host) or
                    not 0 <= anchor_age <= 1000):
                return False
            # The wire expiry is relative to the earlier MCU clock advertisement.
            frame_age = 0 if sample['captured_ms'] is None else write_now - sample['captured_ms']
            if frame_age < 0 or anchor_age + frame_age >= TTL_MS:
                self._vision_expired_skips += 1
                return False
            packet = encode_observation(host=self._vision_host, sequence=self._vision_sequence,
                advertised_ms=data['now_ms'], label=sample['label'], confidence=sample['confidence'],
                frame_age_ms=frame_age)
            try:
                count = self._serial.write(packet)
                if type(count) is not int or not 0 <= count <= len(packet):
                    raise ValueError('Invalid serial write count')
                self._vision_bytes += count
                if count != len(packet):
                    raise ValueError('Partial observation write')
            except (OSError, ValueError) as error:
                # Do not retry a partial frame or close a link owned by the reader.
                self._vision_fault = type(error).__name__
                return False
            self._vision_messages += 1
            self._vision_written.append((self._vision_sequence, sample['label']))
            self._vision_sequence = (self._vision_sequence + 1) & 0xFFFFFFFF
            if sample['captured_ms'] is not None:
                self._vision_last_capture = sample['captured_ms']
            self._vision_last_label = sample['label']
            return True

    def snapshot(self, now_ms):
        result = super().snapshot(now_ms)
        with self._lock:
            capability = result.get('visionLink') or {}
            data = capability.get('data') or {}
            known = dict(self._vision_written)
            acknowledged = (data.get('seq') if capability.get('fresh') is True and
                data.get('host') == self._vision_host and data.get('model') == MODEL_SHA256 and
                data.get('seq') in known else None)
            result.update(transport='h743_native_usb_cdc_observation_uplink',
                applicationBytesTransmitted=self._vision_bytes,
                visionUplink=dict(enabled=True, messagesWritten=self._vision_messages,
                    expiredBeforeWriteSkips=self._vision_expired_skips,
                    host=self._vision_host, fault=self._vision_fault,
                    acknowledgedSequence=acknowledged,
                    acknowledgedObservationFresh=acknowledged is not None and capability.get('observationFresh') is True,
                    writeExceptionMayHaveUncountedBytes=self._vision_fault is not None,
                    actuatorCommandsSupported=False))
        return result
