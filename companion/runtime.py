"""Run the existing local model without sending actuator commands."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

from .capture import LiveFrameReader
from .state import Detection, LABELS, TemporalState


def default_weights() -> Path:
    desktop = Path(__file__).resolve().parents[2]
    return desktop / "projects/infant_emotion/runs/detect/baby_state_yolo11s_clean/weights/best.pt"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_kind(source: str) -> str:
    if source.isdecimal():
        return "camera"
    if urlsplit(source).scheme.lower() in ("http", "https", "rtsp"):
        return "network"
    return "file"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local baby-state observation companion (no actuator control)")
    parser.add_argument("--source", help="image, video, camera index, or camera stream URL")
    parser.add_argument("--telemetry-only", action="store_true",
                        help="observe the explicit serial endpoint without loading a model or camera")
    parser.add_argument("--weights", type=Path, default=default_weights())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--confirm-ms", type=int, default=1000)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--max-gap-ms", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means until EOF or interruption")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="cooperative session limit in seconds; 0 disables, cleanup may add time")
    parser.add_argument("--fps", type=float, default=5.0, help="maximum processed frames per second")
    parser.add_argument("--jsonl", type=Path, help="optional metadata-only log; no images are saved")
    parser.add_argument("--serial-port", help="explicit port for the selected serial transport; read-only by default")
    parser.add_argument("--serial-transport", choices=("uart-adapter", "native-usb"), default="uart-adapter")
    parser.add_argument("--vision-uplink", action="store_true",
                        help="opt-in observation-only USB OUT to a matching new H743 firmware; no actuator commands")
    parser.add_argument("--usb-serial-number")
    parser.add_argument("--usb-location")
    parser.add_argument("--native-board-confirmed", action="store_true")
    parser.add_argument("--isolated-usb-confirmed", action="store_true")
    parser.add_argument("--voice-rx-harness-confirmed", action="store_true")
    parser.add_argument("--actuators-disconnected-confirmed", action="store_true")
    parser.add_argument("--separate-usb-power-confirmed", action="store_true")
    parser.add_argument("--dht11-3v3-harness-confirmed", action="store_true",
                        help="only DHT11 on MCU 3V3/GND/PB0, DATA voltage checked, no external supply/UART")
    parser.add_argument("--voice-and-dht11-harness-confirmed", action="store_true",
                        help="supervised powered voice TX1 to PD6 plus MCU-powered 3V3 DHT11 on PB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--serial-reconnect-seconds", type=float, default=0,
                        help="opt-in 0..300 second retry window for the same serial port; 0 disables")
    parser.add_argument("--retries", type=int, default=3, help="bounded live-source reconnect attempts")
    parser.add_argument("--reconnect-seconds", type=float, default=0,
                        help="0 uses --retries; otherwise retry within this 0..300 second window")
    parser.add_argument("--stream-read-timeout-ms", type=int, default=3000,
                        help="network video read timeout in milliseconds, 250..10000; freshness is separate")
    parser.add_argument("--network-reader", choices=("ffmpeg", "http-mjpeg", "http-snapshot"), default="ffmpeg",
                        help="opt-in multipart or single-JPEG HTTP reader; default keeps FFmpeg")
    parser.add_argument("--snapshot-fallback-url",
                        help="opt-in same-camera JPEG URL after MJPEG failure; disabled by default")
    parser.add_argument("--exit-on-video-loss", action="store_true",
                        help="close a combined observer when the live-video retry budget is exhausted")
    parser.add_argument("--dashboard-port", type=int, default=0, help="localhost read-only dashboard; 0 disables")
    args = parser.parse_args(argv)
    if args.telemetry_only:
        if args.source is not None or not args.serial_port:
            parser.error("telemetry-only requires an explicit serial port and no video source")
    elif not args.source:
        parser.error("a video source or --telemetry-only is required")
    if args.max_frames < 0 or args.retries < 0 or not math.isfinite(args.fps) or not 0 < args.fps <= 60:
        parser.error("invalid frame limit, retry count, or FPS")
    if not math.isfinite(args.max_seconds) or not 0 <= args.max_seconds <= 86400:
        parser.error("max seconds must be finite and in 0..86400")
    if args.exit_on_video_loss and (args.telemetry_only or not args.serial_port
                                   or source_kind(args.source) == "file"):
        parser.error("exit-on-video-loss requires a live video source and a serial port")
    if not math.isfinite(args.reconnect_seconds) or not 0 <= args.reconnect_seconds <= 300:
        parser.error("reconnect seconds must be finite and in 0..300")
    if not 250 <= args.stream_read_timeout_ms <= 10000:
        parser.error("stream read timeout must be in 250..10000 milliseconds")
    if args.network_reader in ("http-mjpeg", "http-snapshot") and (args.telemetry_only or not args.source
            or urlsplit(args.source).scheme.lower() not in ("http", "https")
            or urlsplit(args.source).username is not None or urlsplit(args.source).password is not None
            or urlsplit(args.source).fragment):
        parser.error("HTTP camera readers require an HTTP(S) source without userinfo or fragment")
    if args.network_reader == 'http-snapshot':
        try:
            from .http_snapshot import snapshot_url
            snapshot_url(args.source)
        except ValueError as error:
            parser.error(str(error))
    if args.snapshot_fallback_url:
        try:
            from .http_snapshot import snapshot_url
            fallback = snapshot_url(args.snapshot_fallback_url)
            source = snapshot_url(args.source or '')
            if (args.network_reader != 'http-mjpeg' or args.telemetry_only
                    or fallback.scheme.lower() != source.scheme.lower()
                    or fallback.hostname != source.hostname
                    or args.snapshot_fallback_url == args.source):
                raise ValueError('Snapshot fallback requires a distinct same-host URL and http-mjpeg')
        except ValueError as error:
            parser.error(str(error))
    if args.baud <= 0:
        parser.error("baud must be positive")
    if args.vision_uplink and (args.serial_transport != 'native-usb' or args.telemetry_only or
            not args.source or source_kind(args.source) != 'network' or args.max_gap_ms > 1500 or args.fps > 10):
        parser.error('vision uplink requires native USB, network video, max gap <=1500ms and FPS <=10')
    if not math.isfinite(args.serial_reconnect_seconds) or not 0 <= args.serial_reconnect_seconds <= 300:
        parser.error("serial reconnect seconds must be finite and in 0..300")
    if args.serial_reconnect_seconds and not args.serial_port:
        parser.error("serial reconnect requires an explicit serial port")
    if args.serial_transport == "native-usb":
        from .native_usb import native_wiring_scope
        try:
            native_wiring_scope(native_board_confirmed=args.native_board_confirmed,
                isolated_usb_confirmed=args.isolated_usb_confirmed,
                voice_rx_harness_confirmed=args.voice_rx_harness_confirmed,
                actuators_disconnected_confirmed=args.actuators_disconnected_confirmed,
                separate_usb_power_confirmed=args.separate_usb_power_confirmed,
                dht11_3v3_harness_confirmed=args.dht11_3v3_harness_confirmed,
                voice_and_dht11_harness_confirmed=args.voice_and_dht11_harness_confirmed)
        except ValueError as error:
            parser.error(str(error))
        if any(not value or not value.strip() for value in
               (args.serial_port, args.usb_serial_number, args.usb_location)):
            parser.error("native USB requires endpoint identity")
    elif any((args.usb_serial_number is not None, args.usb_location is not None,
              args.native_board_confirmed, args.isolated_usb_confirmed, args.voice_rx_harness_confirmed,
              args.actuators_disconnected_confirmed, args.separate_usb_power_confirmed,
              args.dht11_3v3_harness_confirmed, args.voice_and_dht11_harness_confirmed)):
        parser.error("native USB identity/confirmation options require --serial-transport native-usb")
    if not 0 <= args.dashboard_port <= 65535:
        parser.error("dashboard port must be in 0..65535")
    try:
        TemporalState(args.confidence, args.confirm_ms, args.min_frames, args.max_gap_ms)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    temporal = TemporalState(args.confidence, args.confirm_ms, args.min_frames, args.max_gap_ms)
    kind = "serial_only" if args.telemetry_only else source_kind(args.source)
    sequence = 0
    frame_count = 0
    telemetry = None
    capture = None
    log = None
    model_id = None
    dashboard = None
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else None
    video_state = "not_requested" if args.telemetry_only else "connecting"
    reconnect_count = 0
    stale_frames_skipped = 0
    termination_reason = None
    active_network_reader = args.network_reader

    def time_expired() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def wait(seconds: float, *, refresh: bool = False) -> None:
        if deadline is None and not refresh:
            time.sleep(seconds)
            return
        end = time.monotonic() + seconds
        if deadline is not None:
            end = min(end, deadline)
        while time.monotonic() < end:
            if refresh:
                emit()
            time.sleep(min(1.0 / args.fps, max(0.0, end - time.monotonic())))

    def emit(detections: list[Detection] | None = None, inference_ms: float | None = None,
             captured_ms: int | None = None) -> None:
        nonlocal sequence
        now = time.monotonic_ns() // 1000000
        if captured_ms is not None and now - captured_ms > args.max_gap_ms and video_state == "streaming":
            temporal.invalidate("stale")
            detections = None
        record = {
            "type": "device_observation" if args.telemetry_only else "vision_observation",
            "schemaVersion": 1,
            "sequence": sequence,
            "timestampMs": now,
            "frameCapturedMs": captured_ms,
            "frameAgeMs": None if captured_ms is None else max(0, now - captured_ms),
            "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "sourceKind": kind,
            "video": {"connectionState": video_state, "reconnectAttempts": reconnect_count,
                      "staleFramesSkipped": stale_frames_skipped,
                      "networkReader": active_network_reader if kind == "network" else None,
                      "snapshotFallbackEnabled": bool(args.snapshot_fallback_url),
                      "frameTimestampBasis": ({'http-mjpeg': 'multipart_start_received',
                                               'http-snapshot': 'snapshot_request_started'}.get(
                                                   active_network_reader, 'decode_completed')) if kind == "network" else None,
                      "openTimeoutMs": 3000 if kind == "network" else None,
                      "readTimeoutMs": args.stream_read_timeout_ms if kind == "network" else None},
            "terminationReason": termination_reason,
            "model": None if args.telemetry_only else {"name": args.weights.stem, "sha256": model_id},
            "observation": temporal.expire(now).to_dict(),
            "detections": [asdict(item) for item in detections or []],
            "inferenceMs": inference_ms,
            "device": telemetry.snapshot(now) if telemetry is not None else None,
            "actuatorControlEnabled": False,
        }
        if args.vision_uplink and telemetry is not None:
            telemetry.publish_vision(record)
            record['device'] = telemetry.snapshot(time.monotonic_ns() // 1000000)
        sequence += 1
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        print(line, flush=True)
        if log is not None:
            log.write(line + "\n")
            log.flush()
        if dashboard is not None:
            dashboard.publish(record)

    try:
        with ExitStack() as resources:
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                log = args.jsonl.open("x", encoding="utf-8")
            if not args.telemetry_only:
                if not args.weights.is_file() or args.weights.suffix != ".pt":
                    raise ValueError("provide a trusted local .pt weights file")
                model_id = file_digest(args.weights)
                if args.vision_uplink:
                    from .vision_protocol import MODEL_SHA256
                    if model_id != MODEL_SHA256:
                        raise ValueError('Vision uplink requires the pinned model shared by the H743 image')
                with redirect_stdout(sys.stderr):
                    from ultralytics import YOLO
                    import cv2
                    import numpy as np

                    model = YOLO(str(args.weights.resolve()))
                if model.task != "detect" or set(model.names.values()) != LABELS:
                    raise ValueError("weights must contain exactly Cry, Happy, Neutral and Sleep detection classes")
            if time_expired():
                termination_reason = "time_limit"
                temporal.invalidate("stopped")
                emit()
                return 2
            if args.dashboard_port:
                from .dashboard import DashboardServer
                dashboard = DashboardServer(args.dashboard_port, args.max_gap_ms)
                dashboard.start()
                resources.callback(dashboard.close)
                print(f"Read-only dashboard: {dashboard.url}", file=sys.stderr, flush=True)
            if args.serial_port:
                from .telemetry import SerialTelemetry

                options = {"reconnect_seconds": args.serial_reconnect_seconds} if args.serial_reconnect_seconds else {}
                if args.serial_transport == "native-usb":
                    from .native_usb import NativeUsbTelemetry
                    if args.voice_rx_harness_confirmed:
                        options.update(voice_rx_harness_confirmed=True,
                                       actuators_disconnected_confirmed=args.actuators_disconnected_confirmed,
                                       separate_usb_power_confirmed=args.separate_usb_power_confirmed)
                    elif args.dht11_3v3_harness_confirmed:
                        options.update(dht11_3v3_harness_confirmed=True,
                                       actuators_disconnected_confirmed=args.actuators_disconnected_confirmed)
                    elif args.voice_and_dht11_harness_confirmed:
                        options.update(voice_and_dht11_harness_confirmed=True,
                                       actuators_disconnected_confirmed=args.actuators_disconnected_confirmed,
                                       separate_usb_power_confirmed=args.separate_usb_power_confirmed)
                    native_class = NativeUsbTelemetry
                    if args.vision_uplink:
                        from .vision_uplink import NativeVisionTelemetry
                        native_class = NativeVisionTelemetry
                        options['observation_uplink_confirmed'] = True
                    telemetry = native_class(
                        args.serial_port, args.baud, expected_serial=args.usb_serial_number,
                        expected_location=args.usb_location, native_board_confirmed=args.native_board_confirmed,
                        isolated_usb_confirmed=args.isolated_usb_confirmed, **options)
                else:
                    telemetry = SerialTelemetry(args.serial_port, args.baud, **options)
                resources.callback(telemetry.close)

            if args.telemetry_only:
                temporal.invalidate("telemetry_only")
                while (not args.max_frames or frame_count < args.max_frames) and not time_expired():
                    started = time.monotonic()
                    emit()
                    frame_count += 1
                    if not args.max_frames or frame_count < args.max_frames:
                        wait(max(0.0, 1.0 / args.fps - (time.monotonic() - started)))
                termination_reason = "time_limit" if time_expired() else "frame_limit"
                temporal.invalidate("stopped")
                emit()
                return 0

            def open_capture():
                source = int(args.source) if kind == "camera" else args.source
                if kind == "network":
                    if active_network_reader in ('http-mjpeg', 'http-snapshot'):
                        if active_network_reader == 'http-snapshot':
                            from .http_snapshot import HttpSnapshotCapture
                            backend = HttpSnapshotCapture(args.snapshot_fallback_url or args.source,
                                read_timeout_ms=args.stream_read_timeout_ms,
                                interval_ms=max(200, min(10000, round(1000 / args.fps))))
                        else:
                            from .http_mjpeg import HttpMjpegCapture
                            backend = HttpMjpegCapture(args.source, read_timeout_ms=args.stream_read_timeout_ms)
                        return LiveFrameReader(backend, shutdown_timeout=backend.shutdown_timeout)
                    backend = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, args.stream_read_timeout_ms,
                    ])
                    # Let the bounded backend read finish before joining its owning thread.
                    return LiveFrameReader(backend, shutdown_timeout=args.stream_read_timeout_ms / 1000 + 1)
                else:
                    backend = cv2.VideoCapture(source)
                return LiveFrameReader(backend) if kind != "file" else backend

            image = None
            if kind == "file":
                path = Path(args.source)
                if not path.is_file():
                    raise ValueError("source file does not exist")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("image could not be decoded")
            if image is None and not time_expired():
                capture = open_capture()
            retries = 0
            reconnect_deadline = None

            def exhausted_video() -> int:
                nonlocal video_state, termination_reason
                video_state = "retry_exhausted"
                temporal.invalidate("stream_lost")
                emit()
                if telemetry is not None and not args.exit_on_video_loss:
                    # Video failure must not hide independently received MCU faults.
                    while not time_expired():
                        wait(1.0 / args.fps)
                        if not time_expired():
                            emit()
                    termination_reason = "time_limit"
                    temporal.invalidate("stopped")
                    emit()
                else:
                    termination_reason = "video_retry_exhausted"
                    emit()
                return 2

            while (not args.max_frames or frame_count < args.max_frames) and not time_expired():
                cycle_started = time.monotonic()
                captured_ms = None
                if image is not None:
                    frame = image
                else:
                    if kind == "file":
                        ok, frame = capture.read()
                    else:
                        latest = capture.next_frame()
                        if latest is None and not capture.failed:
                            emit()
                            continue
                        ok = latest is not None
                        frame = latest.image if latest is not None else None
                        captured_ms = latest.captured_ms if latest is not None else None
                    if not ok or frame is None:
                        if kind == "file" and frame_count == 0:
                            raise ValueError("video source returned no decoded frames")
                        temporal.invalidate("end_of_stream" if kind == "file" else "stream_lost")
                        video_state = "end_of_stream" if kind == "file" else "reconnecting"
                        emit()
                        if kind == "file":
                            break
                        capture.release()
                        capture = None
                        if args.reconnect_seconds:
                            if reconnect_deadline is None:
                                reconnect_deadline = time.monotonic() + args.reconnect_seconds
                            if time.monotonic() >= reconnect_deadline:
                                return exhausted_video()
                        elif retries >= args.retries:
                            return exhausted_video()
                        if args.snapshot_fallback_url and active_network_reader == 'http-mjpeg':
                            # The old capture has joined and closed before a new endpoint is opened.
                            active_network_reader = 'http-snapshot'
                        retries += 1
                        reconnect_count += 1
                        delay = min(retries, 3)
                        if reconnect_deadline is not None:
                            delay = min(delay, max(0, reconnect_deadline - time.monotonic()))
                        wait(delay, refresh=telemetry is not None)
                        if time_expired():
                            break
                        if reconnect_deadline is not None and time.monotonic() >= reconnect_deadline:
                            return exhausted_video()
                        capture = open_capture()
                        continue
                if captured_ms is not None and time.monotonic_ns() // 1000000 - captured_ms > args.max_gap_ms:
                    stale_frames_skipped += 1
                    video_state = "streaming"
                    temporal.invalidate("stale")
                    emit(captured_ms=captured_ms)
                    wait(max(0.0, 1.0 / args.fps - (time.monotonic() - cycle_started)))
                    continue
                started = time.monotonic()
                with redirect_stdout(sys.stderr):
                    result = model.predict(frame, conf=0.05, iou=0.55, imgsz=640,
                                           device=args.device, verbose=False)[0]
                elapsed = round((time.monotonic() - started) * 1000, 3)
                if kind != "file" and capture.failed:
                    video_state = "reconnecting"
                    temporal.invalidate("stream_lost")
                    emit(inference_ms=elapsed, captured_ms=captured_ms)
                    continue
                detections = []
                if result.boxes is not None:
                    for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(),
                                                         result.boxes.conf.cpu().tolist(),
                                                         result.boxes.cls.cpu().tolist()):
                        detections.append(Detection(result.names[int(class_id)], float(confidence), tuple(xyxy)))
                observed_ms = time.monotonic_ns() // 1000000 if captured_ms is None else captured_ms
                temporal.update(detections, observed_ms)
                video_state = "streaming"
                emit(detections, elapsed, captured_ms)
                if dashboard is not None:
                    dashboard.publish_frame(frame, detections, observed_ms, args.confidence)
                retries = 0
                reconnect_deadline = None
                frame_count += 1
                if image is not None:
                    break
                wait(max(0.0, 1.0 / args.fps - (time.monotonic() - cycle_started)))
            termination_reason = "time_limit" if time_expired() else (
                "end_of_stream" if video_state == "end_of_stream" else "frame_limit")
            temporal.invalidate("stopped")
            if video_state == "streaming":
                video_state = "closed"
            emit()
            return 2 if time_expired() and frame_count == 0 else 0
    except KeyboardInterrupt:
        termination_reason = "interrupted"
        temporal.invalidate("interrupted")
        if log is not None and log.closed:
            log = None
        emit()
        return 130
    except Exception as error:
        termination_reason = "runtime_error"
        # Do not put stream credentials, local paths, or frames into error logs.
        temporal.invalidate("runtime_error")
        if log is not None and log.closed:
            log = None
        emit()
        print(f"Companion failed ({type(error).__name__}); check local model/source configuration.", file=sys.stderr)
        return 2
    finally:
        cleanup_failed = False
        if capture is not None:
            try:
                capture.release()
            except Exception:
                cleanup_failed = True
                termination_reason = "runtime_error"
                temporal.invalidate("runtime_error")
                emit()
                print("Capture cleanup failed; the backend may not support bounded reads.", file=sys.stderr)
        if log is not None:
            log.close()
        if cleanup_failed:
            return 2
