"""Triage saved still-image errors without inference, tuning, or device access."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from companion.dataset_audit import NAMES, file_hash, require
from companion.runtime import parse_args as runtime_args
from companion.state import Detection, intersection_over_union
from companion.static_evaluation import MATCH_IOU, no_network, score_image, summarize, truth_boxes


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def verify_inputs(expected, runtime_archive=None):
    bound, drift = {}, []
    runtime_path = (ROOT / 'companion/runtime.py').resolve()
    for name, digest in expected.items():
        path = Path(name).resolve()
        actual = file_hash(path)
        bound[str(path)] = actual
        if actual == digest:
            continue
        require(path == runtime_path and runtime_archive is not None,
                'Changed input requires investigation: ' + str(path))
        archive = Path(runtime_archive).resolve()
        require(file_hash(archive) == digest, 'Runtime archive does not match historical source')
        bound[str(archive)] = digest
        drift.append(dict(originalPath=str(path), originalSha256=digest,
                          currentSha256=actual, matchingArchive=str(archive),
                          archiveExecuted=False))
    return bound, drift


def error_cases(row, confidence, matching_iou):
    truth = [Detection(**item) for item in row['groundTruth']]
    predicted = [Detection(**item) for item in row['predictions']]
    accepted = [Detection(**item) for item in row['acceptedDetections']]
    matched_truth = {pair[0] for pair in row['classAwareMatches']}
    matched_predicted = {pair[1] for pair in row['classAwareMatches']}
    cases = []

    def base(kind, index, item, group, overlaps):
        return dict(sequence=row['sequence'], image=row['image'], kind=kind, index=index,
                    label=item.label, group=group, xyxy=item.xyxy, confidence=item.confidence,
                    groundTruthCount=len(truth), savedPredictionCount=len(predicted),
                    gateReason=row['singleFrameGate']['reason'],
                    maxIoU=max((iou for _, iou in overlaps), default=0.0))

    for index, ground in enumerate(truth):
        if index in matched_truth:
            continue
        overlaps = [(item, intersection_over_union(ground, item)) for item in predicted]
        localized = [item for item, iou in overlaps if iou >= matching_iou]
        correct = [item.confidence for item in localized if item.label == ground.label]
        wrong = [item.confidence for item in localized if item.label != ground.label]
        # Ordered evidence categories, not claims about the underlying cause.
        if any(value >= confidence for value in correct):
            group = 'assignment_competition'
        elif any(value >= confidence for value in wrong):
            group = 'accepted_wrong_class'
        elif correct:
            group = 'below_threshold_correct_class'
        elif wrong:
            group = 'below_threshold_wrong_class'
        elif not predicted:
            group = 'no_saved_predictions'
        elif any(iou > 0 for _, iou in overlaps):
            group = 'insufficient_iou'
        else:
            group = 'no_overlap'
        case = base('FN', index, ground, group, overlaps)
        case.update(bestCorrectLocalizedConfidence=max(correct, default=None),
                    bestWrongLocalizedConfidence=max(wrong, default=None),
                    wrongClassOutranksCorrect=bool(correct and wrong and max(wrong) > max(correct)))
        cases.append(case)

    for index, detection in enumerate(accepted):
        if index in matched_predicted:
            continue
        overlaps = [(item, intersection_over_union(item, detection)) for item in truth]
        localized = [item for item, iou in overlaps if iou >= matching_iou]
        if any(item.label == detection.label for item in localized):
            group = 'duplicate_or_assignment_competition'
        elif localized:
            group = 'wrong_class_overlap'
        elif not truth:
            group = 'background_detection'
        elif any(iou > 0 for _, iou in overlaps):
            group = 'insufficient_iou'
        else:
            group = 'no_overlap'
        cases.append(base('FP', index, detection, group, overlaps))
    require(sum(item['kind'] == 'FN' for item in cases) == row['fn'], 'FN accounting mismatch')
    require(sum(item['kind'] == 'FP' for item in cases) == row['fp'], 'FP accounting mismatch')
    return cases


def analyze(evaluation, output, runtime_archive=None):
    evaluation, output = evaluation.resolve(), output.resolve()
    report_path, predictions_path = evaluation / 'report.json', evaluation / 'predictions.jsonl'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    require(report['scope'] == 'audited_test_stills_at_deployed_candidate_gate_not_live_acceptance',
            'Only the saved audited still evaluation is supported')
    require(file_hash(predictions_path) == report['predictionsSha256'], 'Predictions changed')
    require(report['matchingIoU'] == MATCH_IOU and report['candidateConfidence'] ==
            runtime_args(['--source', 'unused-local-still.png']).confidence, 'Scoring defaults changed')
    bound, drift = verify_inputs({**report['inputs'], **report['datasetInputFiles']}, runtime_archive)
    for path in (report_path, predictions_path, Path(__file__)):
        bound[str(path.resolve())] = file_hash(path)
    indexes = [Path(path) for path in report['inputs'] if Path(path).name == 'images.jsonl']
    audits = [Path(path) for path in report['inputs'] if Path(path).name == 'audit.json']
    require(len(indexes) == len(audits) == 1, 'Exactly one audited index required')
    audit = json.loads(audits[0].read_text(encoding='utf-8'))
    dataset_root = Path(audit['root']).resolve()
    require(not output.exists() and not output.is_relative_to(dataset_root) and
            not output.is_relative_to(evaluation) and not output.is_relative_to(audits[0].parent),
            'Use a new output directory outside the dataset, audit, and saved evaluation')
    index = [json.loads(line) for line in indexes[0].read_text(encoding='utf-8').splitlines()]
    index = [item for item in index if item['split'] == 'test']
    rows = [json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines()]
    require(len(rows) == len(index) == report['summary']['images'], 'Image inventory changed')
    cases = []
    for sequence, (row, source) in enumerate(zip(rows, index)):
        require(row['sequence'] == sequence and row['image'] == source['image'] and
                row['imageSha256'] == source['imageSha256'], 'Image/index identity mismatch')
        truth = truth_boxes(source['boxes'], source['width'], source['height'])
        require(row['groundTruth'] == normalized([asdict(item) for item in truth]), 'Saved labels differ from audit')
        predicted = [Detection(**item) for item in row['predictions']]
        score = normalized(score_image(truth, predicted))
        require(all(row[key] == value for key, value in score.items()), 'Saved scores differ from replay')
        for case in error_cases(row, report['candidateConfidence'], report['matchingIoU']):
            x1, y1, x2, y2 = case['xyxy']
            case.update(width=source['width'], height=source['height'],
                        boxAreaFraction=(x2-x1)*(y2-y1)/(source['width']*source['height']))
            cases.append(case)
    require(summarize(rows) == report['summary'], 'Saved summary differs from replay')
    require(all(file_hash(path) == digest for path, digest in bound.items()), 'Inputs changed during analysis')
    breakdown = {kind: dict(Counter(item['group'] for item in cases if item['kind'] == kind))
                 for kind in ('FN', 'FP')}
    result = dict(scope='saved_still_error_triage_not_new_inference_or_live_acceptance',
                  sourceReport=str(report_path), datasetRoot=str(dataset_root),
                  savedMetrics=report['summary'], breakdown=breakdown,
                  falseNegativesByClass={name: dict(Counter(item['group'] for item in cases
                      if item['kind'] == 'FN' and item['label'] == name)) for name in NAMES},
                  lowConfidenceCorrectWithHigherWrongClass=sum(item['kind'] == 'FN' and
                      item['group'] == 'below_threshold_correct_class' and
                      item['wrongClassOutranksCorrect'] for item in cases),
                  gateRejectedImages=[dict(sequence=row['sequence'], image=row['image'],
                      groundTruthCount=row['groundTruthCount'], reason=row['singleFrameGate']['reason'])
                      for row in rows if row['singleFrameGate']['candidate'] is None],
                  cases=cases, sourceDrift=drift, checkedInputHashes=bound,
                  replayedImages=len(rows), allSavedScoresMatchCurrentScoringDefaults=True,
                  historicalRuntimeIdentityClaimedForCurrentCode=False,
                  newInferencePerformed=False, modelOrThresholdOrDatasetModified=False,
                  hardwareAccessPerformed=False, liveOrClinicalAcceptance=False,
                  limitations=['Groups describe saved post-NMS evidence, not root causes.',
                      'No saved prediction does not mean no output below the 0.05 prediction floor.',
                      'An FN and an FP may refer to the same wrong-class box; do not add them as image counts.',
                      'Test images must not be used to select thresholds or training augmentations.',
                      'No background images; subject/video split independence is not established.',
                      'Still-image matching is not COCO AP or temporal-state acceptance.'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    columns = list(dict.fromkeys(key for item in cases for key in item))
    with (output / 'error-cases.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cases)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--runtime-archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    with no_network():
        result = analyze(args.evaluation, args.output, args.runtime_archive)
    print(json.dumps({key: result[key] for key in ('replayedImages', 'breakdown', 'falseNegativesByClass',
                                                 'lowConfidenceCorrectWithHigherWrongClass', 'sourceDrift')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
