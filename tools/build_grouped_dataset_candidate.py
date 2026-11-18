"""Build an isolated development dataset with conservative similarity constraints."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from companion.dataset_audit import NAMES, dataset_layout, file_hash, require, require_local_asset
from companion.static_evaluation import no_network
from tools.review_dataset_similarity import run_review, verify_unchanged

RANK = {'train': 0, 'val': 1, 'test': 2}
DIRECTORY = {'train': 'train', 'val': 'valid', 'test': 'test'}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def split_counts(rows, field):
    result = {s: dict(images=0, instances=0, classes={n: 0 for n in NAMES}) for s in RANK}
    for row in rows:
        stats = result[row[field]]
        stats['images'] += 1
        stats['instances'] += len(row['boxes'])
        for box in row['boxes']:
            stats['classes'][NAMES[box[0]]] += 1
    return result


def make_plan(records, pairs, manifest):
    require(records, 'An empty dataset cannot be grouped')
    records = sorted(records, key=lambda r: r['image'])
    index = {r['image']: i for i, r in enumerate(records)}
    require(len(index) == len(records), 'Duplicate image record')
    for row in records:
        require(row['split'] in RANK and row['label'] is not None, 'Explicit split and annotation required')
    manifest_index = {}
    for row in manifest:
        key = {'valid': 'val'}.get(row['split'], row['split']), row['file']
        require(key not in manifest_index, 'Duplicate manifest key')
        manifest_index[key] = row
    by_cluster, by_hint = defaultdict(list), defaultdict(list)
    for row in records:
        key = row['split'], Path(row['image']).stem
        require(key in manifest_index, 'Image absent from manifest')
        original = manifest_index[key]
        by_cluster[original['cluster_id']].append(index[row['image']])
        by_hint[original['source_id']].append(index[row['image']])
    edges = set()
    for group in list(by_cluster.values()) + list(by_hint.values()):
        for member in group[1:]:
            edges.add(tuple(sorted((group[0], member))))
    for pair in pairs:
        require(pair['imageA'] in index and pair['imageB'] in index, 'Candidate references an unknown image')
        a, b = index[pair['imageA']], index[pair['imageB']]
        require(a != b, 'Self-pair is not valid')
        edges.add(tuple(sorted((a, b))))
    ordered_edges = sorted(edges)
    a = np.array([e[0] for e in ordered_edges], dtype=np.int64)
    b = np.array([e[1] for e in ordered_edges], dtype=np.int64)
    graph = coo_matrix((np.ones(len(a), dtype=np.uint8), (a, b)), shape=(len(records), len(records))).tocsr()
    _, components = connected_components(graph, directed=False)
    members = defaultdict(list)
    for i, component in enumerate(components):
        members[int(component)].append(records[i])
    groups = []
    for group in members.values():
        paths = sorted(r['image'] for r in group)
        group_id = hashlib.sha256('\n'.join(paths).encode('utf-8')).hexdigest()
        splits = sorted({r['split'] for r in group}, key=RANK.get)
        # Never send an original validation/test image back into a less protected split.
        destination = max(splits, key=RANK.get)
        groups.append(dict(groupId=group_id, images=paths, originalSplits=splits, assignedSplit=destination,
                           sourceIdentityVerified=False, constraintOnly=True))
    groups.sort(key=lambda g: g['groupId'])
    membership = {name: (i, g) for i, g in enumerate(groups) for name in g['images']}
    plan = []
    for row in records:
        i, group = membership[row['image']]
        identity = row['split'] + '_' + hashlib.sha256(row['image'].encode('utf-8')).hexdigest()[:24]
        prefix = DIRECTORY[group['assignedSplit']]
        original = manifest_index[(row['split'], Path(row['image']).stem)]
        plan.append(dict(originalImage=row['image'], originalLabel=row['label'], originalSplit=row['split'],
                         assignedSplit=group['assignedSplit'], imageSha256=row['imageSha256'],
                         labelSha256=row['labelSha256'], boxes=row['boxes'], groupId=group['groupId'],
                         clusterId=i, originalClusterId=original['cluster_id'], sourceHint=original['source_id'],
                         candidateImage=f'{prefix}/images/{identity}{Path(row["image"]).suffix.lower()}',
                         candidateLabel=f'{prefix}/labels/{identity}.txt'))
    require(len({p['candidateImage'].casefold() for p in plan}) == len(plan), 'Destination image collision')
    require(len({p['candidateLabel'].casefold() for p in plan}) == len(plan), 'Destination label collision')
    require(all(RANK[r['assignedSplit']] >= RANK[r['originalSplit']] for r in plan), 'Held-out backflow')
    assigned = {p['originalImage']: p['assignedSplit'] for p in plan}
    require(all(assigned[p['imageA']] == assigned[p['imageB']] for p in pairs), 'Similarity edge still crosses splits')
    before, after = split_counts(plan, 'originalSplit'), split_counts(plan, 'assignedSplit')
    require(all(after[s]['images'] > 0 for s in RANK), 'Grouping emptied a split; review constraints')
    require(all(after[s]['classes'][c] > 0 for s in RANK for c in NAMES), 'Grouping removed a class; review constraints')
    return plan, groups, dict(before=before, after=after, groups=len(groups),
                             groupsCrossingOriginalSplits=sum(len(g['originalSplits']) > 1 for g in groups),
                             largestGroup=max(len(g['images']) for g in groups),
                             retainedImages=len(plan), removedImages=0,
                             movedImages=sum(p['assignedSplit'] != p['originalSplit'] for p in plan),
                             movement=dict(Counter(p['originalSplit'] + '->' + p['assignedSplit'] for p in plan
                                                   if p['assignedSplit'] != p['originalSplit'])))


def json_file(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def materialize(root, output, plan, attribution=None):
    require(not output.exists(), 'Candidate dataset already exists')
    require(not output.resolve().is_relative_to(root.resolve()), 'Candidate cannot be inside source dataset')
    output.mkdir(parents=True, exist_ok=False)
    for directory in DIRECTORY.values():
        for kind in ('images', 'labels'):
            (output / directory / kind).mkdir(parents=True)
    for row in plan:
        for old, new, digest in [('originalImage', 'candidateImage', 'imageSha256'),
                                 ('originalLabel', 'candidateLabel', 'labelSha256')]:
            source = root / row[old]
            destination = output / row[new]
            require_local_asset(source, root)
            require(destination.resolve().is_relative_to(output.resolve()), 'Candidate output escapes root')
            require(not destination.exists(), 'Candidate destination collision')
            require(file_hash(source) == row[digest], 'Source changed before copy')
            shutil.copyfile(source, destination)
            require(file_hash(destination) == row[digest], 'Candidate copy hash mismatch')
    if attribution is not None:
        require_local_asset(attribution, root)
        shutil.copyfile(attribution, output / 'ATTRIBUTION.txt')
    manifest = [dict(file=Path(p['candidateImage']).stem, split=DIRECTORY[p['assignedSplit']],
                     source_id=p['sourceHint'], cluster_id=p['clusterId'],
                     labels=sorted({NAMES[b[0]] for b in p['boxes']}),
                     original_image=p['originalImage'], original_split=p['originalSplit'],
                     grouping_constraint_not_verified_source=True) for p in plan]
    json_file(output / 'manifest.json', manifest)
    config = dict(train='train/images', val='valid/images', test='test/images', nc=4, names=NAMES)
    with (output / 'data.yaml').open('x', encoding='utf-8') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    with (output / 'DEVELOPMENT_ONLY.txt').open('x', encoding='utf-8') as stream:
        stream.write('DEVELOPMENT CANDIDATE ONLY. Not a final independent holdout.\n'
                     'All original images and labels retained byte-for-byte. Similarity constraints are not identity evidence.\n'
                     'Do not score the existing project-trained model as if this split were unseen.\n'
                     'Fresh initialization and an untouched external holdout are required for final improvement claims.\n'
                     'Use only after the parent candidate.json completion record exists. No automatic deployment.\n')


def build_candidate(config, output):
    config, output = Path(config).resolve(), Path(output).resolve()
    root, _ = dataset_layout(config)
    require(not output.exists(), 'Output must be new; partial builds are not resumed implicitly')
    require(not output.is_relative_to(root), 'Output must remain outside the source dataset')
    source_manifest = root / 'manifest.json'
    require_local_asset(source_manifest, root)
    require(source_manifest.is_file(), 'Source manifest is required')
    attribution = root / 'ATTRIBUTION.txt'
    if attribution.exists():
        require_local_asset(attribution, root)
    attribution_sha = file_hash(attribution) if attribution.is_file() else None
    output.mkdir(parents=True, exist_ok=False)
    print('Scanning the original dataset before grouping...', flush=True)
    before = run_review(config, output / 'source-review', manifest=source_manifest, contact_limit=0)
    manifest_check = before['manifestReview']
    require(not manifest_check['issues'], 'Source manifest has unresolved mismatches')
    records = load_jsonl(output / 'source-review/images.jsonl')
    pairs = load_jsonl(output / 'source-review/pairs.jsonl')
    manifest = json.loads(source_manifest.read_text(encoding='utf-8'))
    plan, groups, counts = make_plan(records, pairs, manifest)
    json_file(output / 'plan.json', plan)
    json_file(output / 'groups.json', groups)
    print(f"Copying {len(plan)} unchanged images and labels; moving {counts['movedImages']} split assignments...", flush=True)
    materialize(root, output / 'dataset', plan, attribution if attribution_sha else None)
    print('Independently scanning the actual candidate files...', flush=True)
    after = run_review(output / 'dataset/data.yaml', output / 'candidate-review',
                       manifest=output / 'dataset/manifest.json', contact_limit=0)
    require(after['summary']['crossSplitCandidates'] == 0, 'Candidate still has known heuristic cross-split matches')
    require(not after['manifestReview']['issues'], 'Candidate manifest mismatch')
    require(not any(after['manifestReview']['crossSplitGroups'].values()), 'Candidate manifest groups cross splits')
    candidate_rows = load_jsonl(output / 'candidate-review/images.jsonl')
    actual = {r['image']: r for r in candidate_rows}
    require(set(actual) == {p['candidateImage'] for p in plan}, 'Candidate image membership mismatch')
    for row in plan:
        saved = actual[row['candidateImage']]
        require(saved['imageSha256'] == row['imageSha256'] and saved['labelSha256'] == row['labelSha256']
                and saved['boxes'] == row['boxes'] and saved['split'] == row['assignedSplit'], 'Candidate differs from plan')
    require(after['summary']['candidatePairs'] == before['summary']['candidatePairs'], 'Unexpected change in similarity pairs')
    verify_unchanged(config, before['sourceFiles'])
    if attribution_sha:
        require(file_hash(attribution) == attribution_sha and
                file_hash(output / 'dataset/ATTRIBUTION.txt') == attribution_sha, 'Attribution changed')
    evidence_paths = [output / 'plan.json', output / 'groups.json',
                      output / 'source-review/review.json', output / 'source-review/pairs.jsonl',
                      output / 'candidate-review/review.json', output / 'candidate-review/pairs.jsonl',
                      output / 'dataset/DEVELOPMENT_ONLY.txt']
    if attribution_sha:
        evidence_paths.append(output / 'dataset/ATTRIBUTION.txt')
    result = dict(schemaVersion=1, scope='isolated_development_dataset_candidate', completedBuild=True,
                  sourceConfig=str(config), candidateConfig=str(output / 'dataset/data.yaml'),
                  assignmentPolicy='existing clusters + filename-source hints + all dHash/pHash candidates; test > val > train',
                  automaticGroupingIsNotSourceVerification=True, counts=counts,
                  beforeCrossSplitCandidates=before['summary']['crossSplitCandidates'],
                  afterCrossSplitCandidates=after['summary']['crossSplitCandidates'],
                  similarityCandidatePairsRetained=after['summary']['candidatePairs'],
                  sourceImagesAndLabelsUnchanged=True, everyImageAndLabelPreservedInCandidate=True,
                  originalHeldOutImagesSentToTraining=0, oldModelValidForUnseenEvaluationOnThisSplit=False,
                  requiresIndependentInitialization=True, untouchedFinalHoldoutStillRequired=True,
                  isApprovedFinalDataset=False, sourceOrSubjectIndependenceVerified=False,
                  detectorWeightsModified=False, trainingPerformed=False, modelDeploymentChanged=False,
                  hardwareAccessPerformed=False, originalAttributionCopied=attribution_sha is not None,
                  evidenceFiles={str(p.relative_to(output)): file_hash(p) for p in evidence_paths},
                  toolHashes={str(p): file_hash(p) for p in
                              [Path(__file__), ROOT / 'tools/review_dataset_similarity.py']})
    json_file(output / 'candidate.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    with no_network():
        result = build_candidate(args.data, args.output)
    print(json.dumps(dict(counts=result['counts'], afterCrossSplitCandidates=result['afterCrossSplitCandidates'],
                          isApprovedFinalDataset=False), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
