"""Compare local weights with cached development validation; never deploy either."""

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from companion.dataset_audit import NAMES, file_hash, require
from companion.runtime import default_weights, parse_args
from companion.state import Detection
from companion.static_evaluation import (MATCH_IOU, PREDICT, effective_settings,
                                        no_network, score_image, summarize, truth_boxes)
from tools.evaluate_validation_candidate import load_validation


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def cached_candidate(path, expected_sha, selected, candidate_sha, bound):
    path = Path(path).resolve()
    require(file_hash(path) == expected_sha, 'Candidate report identity changed')
    report = json.loads(path.read_text(encoding='utf-8'))
    require(report.get('kind') == 'fixed-runtime-gate-development-validation-not-independent-holdout'
            and report.get('split') == 'val' and report.get('completed') is True
            and report.get('modelSha256') == candidate_sha
            and report.get('modelDeployed') is False
            and report.get('testSplitOpenedForInference') is False,
            'Only the completed undeployed development validation may be reused')
    require(report.get('candidateConfidence') == parse_args(['--source', 'unused.png']).confidence
            and report.get('matchingIoU') == MATCH_IOU, 'Scoring gate changed')
    for name, sha in bound.items():
        require(report['inputs'].get(name) == sha, 'Validation provenance differs from cached candidate')
    for name, sha in report['inputs'].items():
        require(file_hash(Path(name)) == sha, 'A cached evaluation input changed')
    predictions = path.parent / 'predictions.jsonl'
    require(file_hash(predictions) == report['predictionsSha256'], 'Cached predictions changed')
    rows = [json.loads(line) for line in predictions.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(rows) == len(selected), 'Cached validation count mismatch')
    for index, (row, item) in enumerate(zip(rows, selected)):
        require(row['sequence'] == index and row['image'] == item['candidateImage']
                and row['imageSha256'] == item['imageSha256'] and row['groupId'] == item['groupId'],
                'Cached validation image order or identity changed')
        truth = [Detection(**box) for box in row['groundTruth']]
        predicted = [Detection(**box) for box in row['predictions']]
        rescored = score_image(truth, predicted)
        require(all(canonical(row[key]) == canonical(value) for key, value in rescored.items()),
                'Cached scores cannot be reproduced with the current scoring code')
    require(summarize(rows) == report['summary'], 'Cached summary is not reproducible')
    return report, rows, {str(path): expected_sha, str(predictions): report['predictionsSha256']}


def comparison_rows(reference, candidate):
    require(len(reference) == len(candidate) and reference, 'Nonempty paired images required')
    result = []
    for old, new in zip(reference, candidate):
        require(all(canonical(old[key]) == canonical(new[key]) for key in
                    ('sequence', 'image', 'imageSha256', 'groupId', 'groundTruth')),
                'Comparison images or ground truths are not identical')
        result.append(dict(sequence=old['sequence'], image=old['image'],
            imageSha256=old['imageSha256'], groupId=old['groupId'],
            referenceTP=old['tp'], candidateTP=new['tp'], tpDelta=new['tp']-old['tp'],
            referenceFP=old['fp'], candidateFP=new['fp'], fpDelta=new['fp']-old['fp'],
            referenceFN=old['fn'], candidateFN=new['fn'], fnDelta=new['fn']-old['fn'],
            referenceCandidate=old['singleFrameGate']['candidate'],
            candidateCandidate=new['singleFrameGate']['candidate']))
    return result


def metric_deltas(reference, candidate):
    def delta(old, new):
        return {key: new[key]-old[key] if old[key] is not None and new[key] is not None else None
                for key in ('tp', 'fp', 'fn', 'precision', 'recall')}
    return dict(micro=delta(reference['micro'], candidate['micro']),
                perClass={name: delta(reference['perClass'][name], candidate['perClass'][name]) for name in NAMES})


def compare(training, candidate_sha, cached_report, cached_report_sha, weights, weights_sha, output):
    workspace, selected, candidate_model, bound = load_validation(training, candidate_sha)
    candidate, candidate_rows, cached_bound = cached_candidate(
        cached_report, cached_report_sha, selected, candidate_sha, bound)
    output, weights = Path(output).resolve(), Path(weights).resolve()
    require(not output.exists() and output.is_relative_to(ROOT / 'build')
            and not output.is_relative_to(Path(training).resolve()), 'Use a new separate workspace output')
    require(weights.suffix == '.pt' and file_hash(weights) == weights_sha
            and weights_sha != candidate_sha, 'Distinct trusted local reference weights required')
    bound.update(cached_bound)
    for path in (weights, default_weights().resolve(), ROOT / 'handoff/demo-config.json',
                 Path(__file__).resolve()):
        bound[str(path)] = file_hash(path)
    require(bound[str(default_weights().resolve())] == weights_sha, 'Reference is not the current model')
    output.mkdir(parents=True)
    (output / 'comparison-tool.py').write_bytes(Path(__file__).read_bytes())
    settings = output / 'model-settings'
    settings.mkdir()
    env = dict(YOLO_CONFIG_DIR=str(settings), YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false',
               CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    scored, timings = [], []
    with patch.dict(os.environ, env), no_network() as network:
        import cv2
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLO
        torch.set_num_threads(4)
        require(Path(ultralytics.utils.USER_CONFIG_DIR).resolve() == settings / 'Ultralytics',
                'Comparison needs isolated model settings')
        model = YOLO(str(weights))
        require(model.task == 'detect' and model.names == dict(enumerate(NAMES)), 'Class mapping mismatch')
        with (output / 'reference-predictions.jsonl').open('x', encoding='utf-8') as stream:
            for index, (item, cached) in enumerate(zip(selected, candidate_rows)):
                path = workspace / item['candidateImage']
                require(file_hash(path) == item['imageSha256'], 'Image changed before inference')
                frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                require(frame is not None, 'Cannot decode validation image')
                height, width = frame.shape[:2]
                truth = truth_boxes(item['boxes'], width, height)
                require(canonical([asdict(box) for box in truth]) == canonical(cached['groundTruth']),
                        'Cached ground truth differs from decoded image and bound labels')
                start = time.perf_counter()
                prediction = model.predict(frame, **PREDICT)[0]
                timings.append((time.perf_counter()-start)*1000)
                require(prediction.names == dict(enumerate(NAMES)), 'Prediction class mapping mismatch')
                predicted = []
                if prediction.boxes is not None:
                    coords, confs, classes = (getattr(prediction.boxes, name).cpu().tolist()
                                             for name in ('xyxy', 'conf', 'cls'))
                    require(len(coords) == len(confs) == len(classes), 'Unequal output lengths')
                    for box, confidence, class_id in zip(coords, confs, classes):
                        require(class_id in range(4) and class_id == int(class_id), 'Invalid class index')
                        predicted.append(Detection(NAMES[int(class_id)], float(confidence), tuple(box)))
                row = score_image(truth, predicted)
                row.update(sequence=index, image=item['candidateImage'], imageSha256=item['imageSha256'],
                    groupId=item['groupId'], groundTruth=[asdict(box) for box in truth],
                    predictions=[asdict(box) for box in predicted], inferenceWallMs=timings[-1])
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                scored.append(row)
                if (index+1) % 32 == 0 or index+1 == len(selected):
                    print('Reference validation: %d/%d' % (index+1, len(selected)), flush=True)
        settings_used = effective_settings(model)
        require(settings_used == candidate['effectivePredictSettings'], 'Effective inference settings differ')
        runtime = dict(torch=str(torch.__version__), ultralytics=ultralytics.__version__,
                       opencv=cv2.__version__, device='cpu', torchThreads=4)
        require(runtime == candidate['runtime'], 'Core inference runtime differs from cached candidate')
        paired = comparison_rows(scored, candidate_rows)
        reference = summarize(scored)
        require(all(file_hash(Path(name)) == sha for name, sha in bound.items()), 'Bound input changed')
        report = dict(schemaVersion=1, kind='paired-development-validation-not-deployment', completed=True,
            split='val', inputs=bound, candidateModelSha256=candidate_sha, referenceModelSha256=weights_sha,
            cachedCandidatePredictionsReused=True, candidateInferenceRepeated=False,
            candidateReport=str(Path(cached_report).resolve()), candidateReportSha256=cached_report_sha,
            confidence=candidate['candidateConfidence'], matchingIoU=MATCH_IOU,
            matchingMethod=candidate['matchingMethod'], effectivePredictSettings=settings_used,
            reference=reference, candidate=candidate['summary'],
            candidateMinusReference=metric_deltas(reference, candidate['summary']),
            pairedImageChanges=dict(tpIncreased=sum(row['tpDelta'] > 0 for row in paired),
                tpDecreased=sum(row['tpDelta'] < 0 for row in paired),
                tpUnchanged=sum(row['tpDelta'] == 0 for row in paired)),
            referenceLatencyMs=dict(firstImageIncludingSetup=timings[0],
                subsequentMedian=float(np.median(timings[1:])), subsequentP95=float(np.percentile(timings[1:], 95))),
            cachedCandidateLatencyMs=candidate['latencyMs'], runtime=runtime, networkGuard=network,
            latencyMeasuredInSameSession=False, completeDependencyEnvironmentsProvenIdentical=False,
            referencePredictionsSha256=file_hash(output / 'reference-predictions.jsonl'),
            candidateValidationUsedForCheckpointSelection=True,
            referenceTrainingExposureToTheseImagesVerified=False,
            sourceOrSubjectIndependenceVerified=False, independentAccuracyQualified=False,
            testSplitOpenedForInference=False, modelDeployed=False, thresholdChanged=False,
            demoConfigChanged=False, physicalDeviceAccessPerformed=False, projectComplete=False)
        with (output / 'paired-changes.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        report['pairedChangesSha256'] = file_hash(output / 'paired-changes.csv')
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--candidate-sha', required=True)
    parser.add_argument('--cached-report', type=Path, required=True)
    parser.add_argument('--cached-report-sha', required=True)
    parser.add_argument('--reference-weights', type=Path, required=True)
    parser.add_argument('--reference-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.training, args.candidate_sha, args.cached_report, args.cached_report_sha,
                     args.reference_weights, args.reference_sha, args.output)
    print(json.dumps(dict(reference=result['reference']['micro'], candidate=result['candidate']['micro'],
                          delta=result['candidateMinusReference']['micro']), allow_nan=False))
