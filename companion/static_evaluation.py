"""Evaluate trusted local weights on audited still images, never on devices."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

from .dataset_audit import (IMAGE_SUFFIXES, NAMES, dataset_layout, file_hash,
                            labels_from_bytes, require, require_local_asset)
from .runtime import default_weights, parse_args as runtime_args
from .state import Detection, TemporalState, intersection_over_union

PREDICT = dict(conf=0.05, iou=0.55, imgsz=640, device='cpu', verbose=False,
               save=False, save_txt=False, save_crop=False, show=False)
MATCH_IOU = 0.50


def assign_pairs(matrix):
    """Maximum cardinality above MATCH_IOU, then maximum total IoU; not COCO AP."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    values = np.asarray(matrix, dtype=float)
    require(values.ndim == 2, 'IoU matrix must be two dimensional')
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'Invalid IoU')
    if not values.shape[0] or not values.shape[1]:
        return []
    eligible = values >= MATCH_IOU
    # One extra match outweighs every possible difference in the IoU tie-breaker.
    weights = np.where(eligible, min(values.shape) + 1 + values, 0)
    left, right = linear_sum_assignment(weights, maximize=True)
    return [(int(i), int(j)) for i, j in zip(left, right) if eligible[i, j]]


def match_boxes(truth, predicted, *, class_aware=True):
    import numpy as np

    values = np.zeros((len(truth), len(predicted)))
    for i, ground in enumerate(truth):
        for j, detection in enumerate(predicted):
            if not class_aware or ground.label == detection.label:
                values[i, j] = intersection_over_union(ground, detection)
    return assign_pairs(values)


def truth_boxes(rows, width, height):
    result = []
    for class_id, x, y, w, h in rows:
        result.append(Detection(NAMES[class_id], 1.0,
                                ((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)))
    require(all(item.valid() for item in result), 'Invalid ground truth geometry')
    return result


def score_image(truth, predicted):
    require(all(item.valid() for item in truth + predicted), 'Invalid detection; evaluation is not silently truncated')
    defaults = runtime_args(['--source', 'unused-local-still.png'])
    gate = TemporalState(defaults.confidence, defaults.confirm_ms, defaults.min_frames, defaults.max_gap_ms)
    observation = gate.update(predicted, 0)
    require(observation.state == 'Unknown', 'A single image must not become a confirmed temporal state')
    accepted = [item for item in predicted if item.confidence >= defaults.confidence]
    matched = match_boxes(truth, accepted)
    localized = match_boxes(truth, accepted, class_aware=False)
    return dict(groundTruthCount=len(truth), predictionCount=len(predicted), acceptedCount=len(accepted),
                acceptedDetections=[asdict(item) for item in accepted],
                singleFrameGate=observation.to_dict(), classAwareMatches=matched,
                localizationMatches=localized,
                tp=len(matched), fp=len(accepted)-len(matched), fn=len(truth)-len(matched))


def summarize(rows):
    totals = Counter()
    by_class = {name: Counter() for name in NAMES}
    confusion = {name: Counter() for name in [*NAMES, 'Spurious']}
    reasons = Counter()
    single = Counter()
    multi_single = 0
    for row in rows:
        truth = [Detection(**item) for item in row['groundTruth']]
        accepted = [Detection(**item) for item in row['acceptedDetections']]
        totals.update({key: row[key] for key in ('tp', 'fp', 'fn')})
        reasons[row['singleFrameGate']['reason']] += 1
        matched = row['classAwareMatches']
        gi, pi = {p[0] for p in matched}, {p[1] for p in matched}
        for i, item in enumerate(truth):
            by_class[item.label]['tp' if i in gi else 'fn'] += 1
        for i, item in enumerate(accepted):
            if i not in pi:
                by_class[item.label]['fp'] += 1
        loc = row['localizationMatches']
        lgi, lpi = {p[0] for p in loc}, {p[1] for p in loc}
        for i, j in loc:
            confusion[truth[i].label][accepted[j].label] += 1
        for i, item in enumerate(truth):
            if i not in lgi:
                confusion[item.label]['Missed'] += 1
        for j, item in enumerate(accepted):
            if j not in lpi:
                confusion['Spurious'][item.label] += 1
        if len(truth) == 1:
            single['images'] += 1
            if row['singleFrameGate']['candidate'] is None:
                single['rejected'] += 1
            elif len(loc) == 0:
                single['candidateNotLocalizedAtIoU50'] += 1
            elif truth[0].label != accepted[0].label:
                single['localizedWrongClassCandidate'] += 1
            else:
                single['correctLabelAndLocalizationCandidate'] += 1
        if len(truth) > 1 and row['singleFrameGate']['candidate'] is not None:
            multi_single += 1

    def metrics(counts):
        tp, fp, fn = (counts[key] for key in ('tp', 'fp', 'fn'))
        return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                    recall=tp/(tp+fn) if tp+fn else None)

    single = {key: single[key] for key in ('images', 'rejected', 'candidateNotLocalizedAtIoU50',
              'localizedWrongClassCandidate', 'correctLabelAndLocalizationCandidate')}
    single['candidateCoverage'] = (single['images']-single['rejected'])/single['images'] if single['images'] else None
    return dict(images=len(rows), groundTruthInstances=sum(q['groundTruthCount'] for q in rows),
                acceptedDetections=sum(q['acceptedCount'] for q in rows), micro=metrics(totals),
                perClass={name: metrics(value) for name, value in by_class.items()},
                singleFrameGateReasons=dict(sorted(reasons.items())), singleGroundTruthImages=single,
                multipleGroundTruthImages=sum(q['groundTruthCount'] > 1 for q in rows),
                multipleGroundTruthImagesWithSingleCandidate=multi_single,
                backgroundImages=sum(q['groundTruthCount'] == 0 for q in rows),
                localizationConfusion={name: {p: value[p] for p in [*NAMES, 'Missed']}
                                       for name, value in confusion.items()})


@contextmanager
def no_network():
    counts = dict(blockedPythonSocketOperations=0)
    def blocked(*args, **kwargs):
        counts['blockedPythonSocketOperations'] += 1
        raise OSError('Network disabled for local evaluation')
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false'))
        for method in ('connect', 'connect_ex', 'sendto', 'sendmsg'):
            if hasattr(socket.socket, method):
                stack.enter_context(patch.object(socket.socket, method, blocked))
        for method in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'create_connection'):
            stack.enter_context(patch.object(socket, method, blocked))
        yield counts


def load_audited_test(data, audit_dir):
    data = Path(data).resolve()
    root, layout = dataset_layout(data)
    report_path, index_path = audit_dir/'audit.json', audit_dir/'images.jsonl'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    require(report['schemaVersion'] == 1 and Path(report['root']).resolve() == root, 'Wrong dataset audit')
    require(report['auditToolSha256'] == file_hash(Path(__file__).with_name('dataset_audit.py')), 'Audit tool changed')
    require(report['labelsAndDecodeValid'] is True and report['noCrossSplitExactDuplicatesFound'] is True, 'Dataset audit not passed')
    require(report['configSha256'] == file_hash(data) == report['inputFiles'][str(data)], 'Dataset config changed')
    for name, digest in report['inputFiles'].items():
        path = Path(name)
        require_local_asset(path, root)
        require(file_hash(path) == digest, 'Audited input changed')
    require(not any(q['image'].startswith(layout['test'].relative_to(root).as_posix()+'/')
                    for q in report['warnings']), 'Resolve test-split annotation/orientation warnings before scoring')
    rows = [json.loads(line) for line in index_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected = [row for row in rows if row['split'] == 'test']
    paths = [(root/row['image']).resolve() for row in selected]
    inventory = {p.resolve() for p in layout['test'].rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    require(len(paths) == len(set(paths)) == report['splits']['test']['images'] and set(paths) == inventory,
            'Test image inventory does not match the audit')
    for path, row in zip(paths, selected):
        require_local_asset(root/row['image'], root)
        require(row['imageSha256'] == report['inputFiles'][str(path)], 'Image index hash mismatch')
        label = layout['test'].parent/'labels'/path.relative_to(layout['test']).with_suffix('.txt')
        if label.exists():
            require_local_asset(label, root)
            raw = label.read_bytes()
            require(row['label'] == label.relative_to(root).as_posix() and
                    row['labelSha256'] == hashlib.sha256(raw).hexdigest() == report['inputFiles'][str(label)] and
                    row['boxes'] == labels_from_bytes(raw), 'Label index does not match the audited annotation')
        else:
            require(row['label'] is None and not row['boxes'], 'Missing label is not an audited background')
    require(sum(len(row['boxes']) for row in selected) == report['splits']['test']['instances'], 'Instance total mismatch')
    return root, selected, report, {str(report_path.resolve()): file_hash(report_path),
                                   str(index_path.resolve()): file_hash(index_path)}


def effective_settings(model):
    args = model.predictor.args
    result = {key: getattr(args, key) for key in
              ('conf', 'iou', 'imgsz', 'rect', 'max_det', 'augment', 'agnostic_nms',
               'classes', 'save', 'save_txt', 'save_crop', 'show')}
    result.update(quantize=getattr(args, 'quantize', None), legacyHalf=getattr(args, 'half', None))
    require(all(result[key] == value for key, value in PREDICT.items()
                if key in result), 'Effective predictor settings drifted')
    return result


def evaluate(data, audit_dir, weights, output):
    root, selected, audit, bound = load_audited_test(data, audit_dir)
    require(weights.is_file() and weights.suffix == '.pt', 'Trusted local .pt weights required')
    output = output.resolve()
    require(not output.exists() and output != root and root not in output.parents, 'New output outside the dataset required')
    bound.update({str(weights.resolve()): file_hash(weights)})
    for name in ['static_evaluation.py', 'runtime.py', 'state.py', 'dataset_audit.py']:
        path = Path(__file__).with_name(name).resolve()
        bound[str(path)] = file_hash(path)
    output.mkdir(parents=True, exist_ok=False)
    settings_dir = output/'tool-settings'
    settings_dir.mkdir()
    rows = []
    latencies = []
    with no_network() as network, patch.dict(os.environ, YOLO_CONFIG_DIR=str(settings_dir)):
        with redirect_stdout(sys.stderr):
            import cv2
            import numpy as np
            import torch
            import ultralytics
            from ultralytics import YOLO
            require(Path(ultralytics.utils.USER_CONFIG_DIR).resolve() == settings_dir/'Ultralytics',
                    'Ultralytics must use the run-local configuration directory')
            model = YOLO(str(weights.resolve()))
        require(model.task == 'detect' and model.names == dict(enumerate(NAMES)), 'Exact model class index mapping required')
        with (output/'predictions.jsonl').open('x', encoding='utf-8') as stream:
            for sequence, item in enumerate(selected):
                path = root/item['image']
                raw = path.read_bytes()
                require(hashlib.sha256(raw).hexdigest() == item['imageSha256'], 'Image changed before inference')
                frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                require(frame is not None and frame.shape[:2] == (item['height'], item['width']), 'Decoded dimensions changed')
                started = time.perf_counter()
                with redirect_stdout(sys.stderr):
                    results = model.predict(frame, **PREDICT)
                latency = (time.perf_counter()-started)*1000
                require(len(results) == 1, 'Expected exactly one result per local still')
                result = results[0]
                require(result.names == dict(enumerate(NAMES)), 'Prediction class indexes changed')
                predicted = []
                if result.boxes is not None:
                    coordinates = result.boxes.xyxy.cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()
                    classes = result.boxes.cls.cpu().tolist()
                    require(len(coordinates) == len(confidences) == len(classes), 'Unequal model output lengths')
                    for xyxy, confidence, class_id in zip(coordinates, confidences, classes):
                        require(class_id in range(4) and int(class_id) == class_id, 'Invalid model class index')
                        predicted.append(Detection(NAMES[int(class_id)], float(confidence), tuple(xyxy)))
                truth = truth_boxes(item['boxes'], item['width'], item['height'])
                row = score_image(truth, predicted)
                row.update(image=item['image'], imageSha256=item['imageSha256'], sequence=sequence,
                           groundTruth=[asdict(box) for box in truth], predictions=[asdict(box) for box in predicted],
                           inferenceWallMs=latency)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                rows.append(row)
                latencies.append(latency)
                if (sequence+1) % 25 == 0 or sequence+1 == len(selected):
                    print(f'Local stills evaluated: {sequence+1}/{len(selected)}', file=sys.stderr, flush=True)
        require(all(file_hash(Path(p)) == sha for p, sha in {**audit['inputFiles'], **bound}.items()),
                'Input/source changed during evaluation; partial output is not a completed report')
        _, layout = dataset_layout(data)
        actual = {str(p.resolve()) for image_dir in layout.values() for parent in [image_dir, image_dir.parent/'labels']
                  for p in parent.rglob('*') if p.is_file() and
                  (p.suffix.lower() in IMAGE_SUFFIXES if parent == image_dir else p.suffix.lower() == '.txt')}
        expected = set(audit['inputFiles']) - {str(Path(data).resolve())}
        require(actual == expected, 'Dataset membership changed during evaluation')
        defaults = runtime_args(['--source', 'unused-local-still.png'])
        report = dict(schemaVersion=1, scope='audited_test_stills_at_deployed_candidate_gate_not_live_acceptance',
                      inputs=bound, datasetInputFiles=audit['inputFiles'], split='test',
                      predictSettings=PREDICT, candidateConfidence=defaults.confidence, matchingIoU=MATCH_IOU,
                      effectivePredictSettings=effective_settings(model),
                      toolSettingsDirectory=str(settings_dir/'Ultralytics'),
                      matchingMethod='maximum_cardinality_then_total_IoU_assignment_not_COCO_AP',
                      summary=summarize(rows),
                      runtime=dict(ultralytics=ultralytics.__version__, torch=str(torch.__version__),
                                   opencv=cv2.__version__, device='cpu', torchThreads=torch.get_num_threads()),
                      latencyMs=dict(firstImageIncludingSetup=latencies[0],
                                     subsequentImagesMedian=float(np.median(latencies[1:])) if len(rows)>1 else None,
                                     subsequentImagesP95=float(np.percentile(latencies[1:], 95)) if len(rows)>1 else None,
                                     excludesImageDecodeAndLogWrite=True, liveCameraFPSQualified=False),
                      networkGuard=network, externalNetworkUsed=False, hardwareAccessPerformed=False,
                      modelOrDatasetModified=False, imagesCopiedOrUploaded=False, temporalConfirmationTested=False,
                      sourceOrSubjectIndependenceVerified=False, clinicalAccuracyQualified=False,
                      actuatorControlEnabled=False,
                      predictionsSha256=file_hash(output/'predictions.jsonl'))
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--weights', type=Path, default=default_weights())
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.data, args.audit, args.weights, args.output)
    except Exception as error:
        print(f'Local static evaluation failed ({type(error).__name__}); any partial output is not acceptance evidence.', file=sys.stderr)
        return 2
    print(json.dumps(report['summary'], allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
