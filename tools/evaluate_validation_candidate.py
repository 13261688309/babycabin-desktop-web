"""Score a completed development model on its validation split, never a holdout."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from companion.dataset_audit import NAMES, file_hash, labels_from_bytes, require
from companion.runtime import default_weights, parse_args
from companion.state import Detection
from companion.static_evaluation import (MATCH_IOU, PREDICT, effective_settings,
    no_network, score_image, summarize, truth_boxes)


def load_validation(training, expected_model_sha):
    training = Path(training).resolve()
    result_path, prepared_path = training / 'result.json', training / 'prepared.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    require(result.get('completed') is True and result.get('modelDeployed') is False
            and result.get('testInferencePerformed') is False
            and result.get('onlyValidationUsedForCheckpointSelection') is True,
            'Completed undeployed development training required')
    receipt_path = prepared_path
    if result.get('mode') == 'development_resumed_training':
        receipt_path = training / 'resume-prepared.json'
        resume = json.loads(receipt_path.read_text(encoding='utf-8'))
        require(resume.get('kind') == 'grouped_development_training_recovery_preparation'
                and Path(resume['output']).resolve() == training
                and resume.get('automaticDeploymentAllowed') is False
                and resume.get('testInferenceAllowed') is False
                and resume.get('freshPrepared') == prepared, 'Recovery preparation boundary changed')
    else:
        require(result.get('mode') == 'development_training', 'Unknown training result mode')
    require(result.get('preparedReceiptSha256') == file_hash(receipt_path), 'Prepared receipt changed')
    require(prepared.get('mode') == 'development_training'
            and prepared.get('automaticDeploymentAllowed') is False
            and prepared.get('testSplitPresentInTrainingConfig') is False, 'Development boundary changed')
    model = training / 'train/weights/best.pt'
    require(len(expected_model_sha) == 64 and all(c in '0123456789abcdef' for c in expected_model_sha),
            'Explicit lowercase SHA256 required')
    require(result.get('checkpointHashes', {}).get('train/weights/best.pt') == expected_model_sha
            == file_hash(model), 'Candidate checkpoint identity mismatch')
    workspace = training / 'workspace'
    config = workspace / 'data.yaml'
    data = yaml.safe_load(config.read_text(encoding='utf-8'))
    require(data.get('names') == NAMES and data.get('val') == 'valid/images'
            and 'test' not in data and Path(data['path']).resolve() == workspace,
            'Only the training workspace validation split is allowed')
    all_rows = prepared['selectedRows']
    require(all(row.get('assignedSplit') in ('train', 'val') for row in all_rows),
            'A test row cannot enter this evaluation')
    rows = [row for row in all_rows if row['assignedSplit'] == 'val']
    require(rows and len(rows) == prepared['selectedCounts']['val'], 'Validation count mismatch')
    require(len({row['candidateImage'] for row in rows}) == len(rows)
            and len({row['candidateLabel'] for row in rows}) == len(rows), 'Duplicate validation rows')
    bound = {str(path): file_hash(path) for path in (result_path, prepared_path, receipt_path, model, config)}
    for row in rows:
        for field, directory, sha in [('candidateImage', 'images', 'imageSha256'),
                                      ('candidateLabel', 'labels', 'labelSha256')]:
            relative = Path(row[field])
            path = (workspace / relative).resolve()
            require(len(relative.parts) == 3 and relative.parts[:2] == ('valid', directory)
                    and not relative.is_absolute() and path.is_relative_to(workspace / 'valid' / directory),
                    'Validation asset escapes its exact split')
            require(file_hash(path) == row[sha], 'Validation asset changed')
            bound[str(path)] = row[sha]
        label = workspace / row['candidateLabel']
        require(labels_from_bytes(label.read_bytes()) == row['boxes'], 'Validation boxes differ from labels')
    for key, directory in [('candidateImage', 'images'), ('candidateLabel', 'labels')]:
        expected = {(workspace / row[key]).resolve() for row in rows}
        actual = {path.resolve() for path in (workspace / 'valid' / directory).rglob('*') if path.is_file()}
        require(actual == expected, 'Validation directory membership changed')
    return workspace, rows, model, bound


def evaluate(training, expected_model_sha, output):
    workspace, selected, weights, bound = load_validation(training, expected_model_sha)
    output = Path(output).resolve()
    require(not output.exists() and output.is_relative_to(ROOT / 'build')
            and not output.is_relative_to(Path(training).resolve()), 'Use a new separate workspace output')
    deployed = default_weights().resolve()
    deployed_before = file_hash(deployed)
    for name in ['companion/static_evaluation.py', 'companion/state.py', 'companion/runtime.py',
                 'companion/dataset_audit.py', 'tools/evaluate_validation_candidate.py']:
        path = ROOT / name
        bound[str(path)] = file_hash(path)
    output.mkdir(parents=True)
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
                'Evaluation requires a fresh isolated model configuration')
        model = YOLO(str(weights))
        require(model.task == 'detect' and model.names == dict(enumerate(NAMES)), 'Class mapping mismatch')
        with (output / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
            for index, item in enumerate(selected):
                path = workspace / item['candidateImage']
                require(file_hash(path) == item['imageSha256'], 'Image changed before inference')
                frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                require(frame is not None, 'Cannot decode a validation image')
                height, width = frame.shape[:2]
                start = time.perf_counter()
                prediction = model.predict(frame, **PREDICT)[0]
                timings.append((time.perf_counter() - start) * 1000)
                require(prediction.names == dict(enumerate(NAMES)), 'Prediction class mapping mismatch')
                predicted = []
                if prediction.boxes is not None:
                    coords = prediction.boxes.xyxy.cpu().tolist()
                    confs = prediction.boxes.conf.cpu().tolist()
                    classes = prediction.boxes.cls.cpu().tolist()
                    require(len(coords) == len(confs) == len(classes), 'Unequal output lengths')
                    for box, confidence, class_id in zip(coords, confs, classes):
                        require(class_id in range(4) and class_id == int(class_id), 'Invalid class index')
                        predicted.append(Detection(NAMES[int(class_id)], float(confidence), tuple(box)))
                truth = truth_boxes(item['boxes'], width, height)
                row = score_image(truth, predicted)
                row.update(sequence=index, image=item['candidateImage'],
                    imageSha256=item['imageSha256'], groupId=item['groupId'],
                    groundTruth=[asdict(box) for box in truth], predictions=[asdict(box) for box in predicted],
                    inferenceWallMs=timings[-1])
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                scored.append(row)
                if (index + 1) % 32 == 0 or index + 1 == len(selected):
                    print('Validation images: %d/%d' % (index + 1, len(selected)), flush=True)
        require(all(file_hash(Path(name)) == sha for name, sha in bound.items()), 'Bound inputs changed')
        require(file_hash(deployed) == deployed_before, 'Deployed model changed')
        report = dict(schemaVersion=1, kind='fixed-runtime-gate-development-validation-not-independent-holdout',
            split='val', completed=True, modelSha256=expected_model_sha, inputs=bound,
            candidateConfidence=parse_args(['--source', 'unused.png']).confidence, matchingIoU=MATCH_IOU,
            matchingMethod='maximum_cardinality_then_total_IoU_assignment_not_COCO_AP',
            effectivePredictSettings=effective_settings(model), summary=summarize(scored),
            latencyMs=dict(firstImageIncludingSetup=timings[0],
                subsequentMedian=float(np.median(timings[1:])),
                subsequentP95=float(np.percentile(timings[1:], 95))),
            runtime=dict(torch=str(torch.__version__), ultralytics=ultralytics.__version__,
                         opencv=cv2.__version__, device='cpu', torchThreads=4),
            networkGuard=network, predictionsSha256=file_hash(output / 'predictions.jsonl'),
            deployedModelSha256=deployed_before, deployedModelUnchanged=True,
            validationUsedForCheckpointSelection=True, independentAccuracyQualified=False,
            testSplitOpenedForInference=False, thresholdChanged=False, modelDeployed=False,
            trainingPerformed=False, imagesSavedOrUploaded=False, temporalConfirmationTested=False,
            physicalDeviceAccessPerformed=False, clinicalAccuracyQualified=False, projectComplete=False)
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--expected-model-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.training, args.expected_model_sha, args.output)
    print(json.dumps(report['summary'], allow_nan=False))
