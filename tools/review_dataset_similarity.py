"""Read-only full-image similarity review; candidates are not verified duplicates."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import io
import json
from pathlib import Path
import sys

import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from companion.dataset_audit import (IMAGE_SUFFIXES, NAMES, audit_dataset,
                                     dataset_layout, file_hash, require)
from companion.static_evaluation import no_network


def features(image):
    result = {}
    for name, picture in [('original', image), ('mirror', ImageOps.mirror(image))]:
        result[name] = dict(
            phash=int(str(imagehash.phash(picture, hash_size=8)), 16),
            dhash=int(str(imagehash.dhash(picture, hash_size=8)), 16),
            rgb=np.asarray(picture.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32),
        )
    return result


def find_candidates(records, hashes, dhash_limit=4, phash_limit=6):
    require(type(dhash_limit) is int and 0 <= dhash_limit <= 64, 'Invalid dHash threshold')
    require(type(phash_limit) is int and 0 <= phash_limit <= 64, 'Invalid pHash threshold')
    require(len(records) == len(hashes), 'Feature count mismatch')
    pairs = []
    counts = Counter()
    for i, left in enumerate(records):
        for j in range(i + 1, len(records)):
            right = records[j]
            counts['allPairsCompared'] += 1
            cross = left['split'] != right['split']
            counts['crossSplitPairsCompared' if cross else 'withinSplitPairsCompared'] += 1
            matches = []
            for transform_a in ('original', 'mirror'):
                a = hashes[i][transform_a]
                for transform_b in ('original', 'mirror'):
                    b = hashes[j][transform_b]
                    dh = (a['dhash'] ^ b['dhash']).bit_count()
                    ph = (a['phash'] ^ b['phash']).bit_count()
                    if dh <= dhash_limit or ph <= phash_limit:
                        matches.append(dict(transformA=transform_a, transformB=transform_b,
                                            dhashDistance=dh, phashDistance=ph,
                                            resizedRgbMae=float(np.abs(a['rgb'] - b['rgb']).mean())))
            if not matches:
                continue
            best = min(matches, key=lambda m: (m['dhashDistance'] + m['phashDistance'], m['resizedRgbMae']))
            baseline = any(m['transformA'] == m['transformB'] == 'original' and
                           m['dhashDistance'] <= dhash_limit for m in matches)
            class_a = sorted({int(box[0]) for box in left['boxes']})
            class_b = sorted({int(box[0]) for box in right['boxes']})
            pairs.append(dict(imageA=left['image'], imageB=right['image'], splitA=left['split'],
                              splitB=right['split'], crossSplit=cross, matches=matches, best=best,
                              baselineDhashCandidate=baseline, classSetA=[NAMES[c] for c in class_a],
                              classSetB=[NAMES[c] for c in class_b], classSetDiffers=class_a != class_b,
                              sameImageBytes=left['imageSha256'] == right['imageSha256'],
                              verifiedDuplicate=False))
    pairs.sort(key=lambda p: (not p['crossSplit'], p['best']['dhashDistance'] + p['best']['phashDistance'],
                              p['best']['resizedRgbMae'], p['imageA'], p['imageB']))
    for index, pair in enumerate(pairs, 1):
        pair['id'] = f'P{index:04d}'
    return pairs, dict(counts)


def review_manifest(path, records):
    if path is None:
        return dict(present=False, sourceOrSubjectIndependenceVerified=False)
    rows = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(rows, list), 'Manifest must be a list')
    actual = {(r['split'], Path(r['image']).stem): r for r in records}
    require(len(actual) == len(records), 'Ambiguous manifest keys')
    seen = set()
    groups = {key: defaultdict(set) for key in ('source_id', 'cluster_id')}
    issues = []
    for row in rows:
        require(isinstance(row, dict), 'Invalid manifest entry')
        split = {'valid': 'val'}.get(row.get('split'), row.get('split'))
        name = row.get('file')
        require(isinstance(name, str), 'Invalid manifest file')
        key = split, name
        require(key not in seen, 'Duplicate manifest entry')
        seen.add(key)
        require(type(row.get('cluster_id')) is int and row['cluster_id'] >= 0, 'Invalid cluster ID')
        require(isinstance(row.get('source_id'), str) and row['source_id'], 'Invalid source hint')
        require(isinstance(row.get('labels'), list) and all(c in NAMES for c in row['labels']), 'Invalid manifest labels')
        if key not in actual:
            issues.append(dict(kind='manifest_entry_without_image', split=split, file=name))
        elif sorted(set(row['labels'])) != sorted({NAMES[b[0]] for b in actual[key]['boxes']}):
            issues.append(dict(kind='manifest_label_set_mismatch', split=split, file=name))
        for field in groups:
            groups[field][row[field]].add(split)
    for split, name in sorted(set(actual) - seen):
        issues.append(dict(kind='image_missing_from_manifest', split=split, file=name))
    return dict(present=True, entries=len(rows), issues=issues,
                groups={field: len(mapping) for field, mapping in groups.items()},
                crossSplitGroups={field: [dict(key=key, splits=sorted(splits)) for key, splits in mapping.items()
                                         if len(splits) > 1] for field, mapping in groups.items()},
                sourceIdsAreFilenameDerivedHints=True, sourceOrSubjectIndependenceVerified=False)


def checked_image(root, row):
    raw = (root / row['image']).read_bytes()
    import hashlib
    require(hashlib.sha256(raw).hexdigest() == row['imageSha256'], 'Image changed during review')
    with Image.open(io.BytesIO(raw)) as picture:
        return ImageOps.exif_transpose(picture).convert('RGB')


def write_contacts(root, records, pairs, output, limit):
    lookup = {row['image']: row for row in records}
    files = []
    for offset in range(0, min(len(pairs), limit), 4):
        page_pairs = pairs[offset: min(offset + 4, limit)]
        page = Image.new('RGB', (840, 300 * len(page_pairs)), 'white')
        draw = ImageDraw.Draw(page)
        for line, pair in enumerate(page_pairs):
            y = line * 300
            best = pair['best']
            draw.text((8, y + 4), f"{pair['id']}  {pair['splitA']} / {pair['splitB']}  A:{best['transformA']} B:{best['transformB']}"
                      f"  d={best['dhashDistance']} p={best['phashDistance']}  CANDIDATE ONLY", fill='black')
            for col, field in enumerate(('imageA', 'imageB')):
                row = lookup[pair[field]]
                picture = checked_image(root, row)
                mirrored = best['transformA' if col == 0 else 'transformB'] == 'mirror'
                if mirrored:
                    picture = ImageOps.mirror(picture)
                picture.thumbnail((400, 240), Image.Resampling.LANCZOS)
                x0, y0 = col * 420 + 8, y + 26
                page.paste(picture, (x0, y0))
                for c, x, cy, w, h in row['boxes']:
                    if mirrored:
                        x = 1 - x
                    bounds = (x0 + (x - w / 2) * picture.width, y0 + (cy - h / 2) * picture.height,
                              x0 + (x + w / 2) * picture.width, y0 + (cy + h / 2) * picture.height)
                    draw.rectangle(bounds, outline='#d42135', width=2)
                    draw.text((max(x0, bounds[0]), max(y0, bounds[1])), NAMES[c], fill='#d42135')
                draw.text((x0, y + 272), Path(row['image']).name[:60], fill='black')
        name = f'contact-{offset // 4 + 1:02d}.png'
        page.save(output / name)
        files.append(name)
    return files


def verify_unchanged(config, expected):
    _, layout = dataset_layout(config)
    inventory = {config.resolve()}
    for images in layout.values():
        inventory.update(p for p in images.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        inventory.update(p for p in (images.parent / 'labels').rglob('*.txt') if p.is_file())
    # An optional manifest is included in the same byte-level snapshot.
    dataset_paths = {Path(p) for p in expected if Path(p).name != 'manifest.json'}
    require(inventory == dataset_paths, 'Dataset membership changed during review')
    for name, digest in expected.items():
        require(file_hash(name) == digest, 'Dataset changed during review')


def run_review(config, output, *, manifest=None, dhash_limit=4, phash_limit=6, contact_limit=32):
    config, output = Path(config).resolve(), Path(output).resolve()
    require(not output.exists(), 'Output must be a new directory')
    root, _ = dataset_layout(config)
    require(not output.is_relative_to(root), 'Output must remain outside the dataset')
    require(type(contact_limit) is int and 0 <= contact_limit <= 200, 'Invalid contact limit')
    require(type(dhash_limit) is int and 0 <= dhash_limit <= 64, 'Invalid dHash threshold')
    require(type(phash_limit) is int and 0 <= phash_limit <= 64, 'Invalid pHash threshold')
    manifest = Path(manifest).resolve() if manifest else None
    require(manifest is None or manifest == root / 'manifest.json', 'Only the local manifest is supported')
    manifest_sha = file_hash(manifest) if manifest else None
    print('Auditing image and annotation bytes...', flush=True)
    audit = audit_dataset(config)
    require(audit['labelsAndDecodeValid'], 'Resolve image/annotation format errors before similarity review')
    rows = audit.pop('records')
    source_files = dict(audit['inputFiles'])
    if manifest:
        source_files[str(manifest)] = manifest_sha
    manifest_review = review_manifest(manifest, rows)
    hashes = []
    for index, row in enumerate(rows, 1):
        hashes.append(features(checked_image(root, row)))
        if index % 500 == 0:
            print(f'Hashed {index}/{len(rows)} images', flush=True)
    pairs, comparisons = find_candidates(rows, hashes, dhash_limit, phash_limit)
    verify_unchanged(config, source_files)
    cross = [p for p in pairs if p['crossSplit']]
    summary = dict(images=len(rows), **comparisons, candidatePairs=len(pairs), crossSplitCandidates=len(cross),
                   baselineDhashCrossSplitCandidates=sum(p['baselineDhashCandidate'] for p in cross),
                   additionalCrossSplitCandidates=sum(not p['baselineDhashCandidate'] for p in cross),
                   crossSplitCandidatesWithDifferentClassSets=sum(p['classSetDiffers'] for p in cross),
                   crossSplitCandidateImages=len({p[k] for p in cross for k in ('imageA', 'imageB')}),
                   annotationWarningsByKind=dict(Counter(w['kind'] for w in audit['warnings'])),
                   annotationWarningImages=len({w['image'] for w in audit['warnings']}))
    output.mkdir(parents=True, exist_ok=False)
    contacts = write_contacts(root, rows, pairs, output, contact_limit)
    verify_unchanged(config, source_files)
    for name, data in [('pairs.jsonl', pairs), ('images.jsonl', rows)]:
        with (output / name).open('x', encoding='utf-8') as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    with (output / 'annotation-audit.json').open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
    result = dict(schemaVersion=2, summary=summary, manifestReview=manifest_review,
                  method=dict(library='ImageHash', hashBits=64, dhashLimit=dhash_limit, phashLimit=phash_limit,
                              candidateRule='dHash <= limit OR pHash <= limit, all four original/mirror A/B combinations',
                              pairMembershipIndependentOfInputOrder=True,
                              comparisonScope='all unordered full-image pairs; no cap; no subject identity recognition',
                              reference='https://github.com/JohannesBuchner/imagehash',
                              limitations=['heuristic candidates need visual/source review',
                                           'not a complete crop/rotation/sequence leakage search',
                                           'same subject or capture session cannot be verified from this scan',
                                           'thresholds select review candidates; not detector confidence thresholds']),
                  sourceFiles=source_files, sourceToolHashes={str(p): file_hash(p) for p in
                      [Path(__file__), ROOT / 'companion/dataset_audit.py', ROOT / 'companion/static_evaluation.py']},
                  versions={name: version(name) for name in ('ImageHash', 'Pillow', 'numpy', 'scipy')},
                  contactSheets=contacts, contactPairCount=min(contact_limit, len(pairs)),
                  datasetUnchangedAfterReview=True, datasetModified=False, modelInferencePerformed=False,
                  hardwareAccessPerformed=False, sourceOrSubjectIndependenceVerified=False,
                  automaticDeletionOrRelabelingPerformed=False, detectorThresholdChanged=False,
                  completionClaim='diagnostic review only; no system or accuracy acceptance')
    with (output / 'review.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--dhash-limit', type=int, default=4)
    parser.add_argument('--phash-limit', type=int, default=6)
    parser.add_argument('--contact-limit', type=int, default=32)
    args = parser.parse_args(argv)
    with no_network():
        result = run_review(args.data, args.output, manifest=args.manifest, dhash_limit=args.dhash_limit,
                            phash_limit=args.phash_limit, contact_limit=args.contact_limit)
    print(json.dumps(result['summary'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
