"""Read-only audit of a local four-state YOLO directory dataset; no model or network."""

from collections import Counter,defaultdict
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct

from PIL import Image,ImageOps
import yaml

NAMES=['Cry','Happy','Neutral','Sleep']
IMAGE_SUFFIXES={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}


def require(condition,message):
    if not condition:raise ValueError(message)


class UniqueSafeLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader,node,deep=False):
    pairs=loader.construct_pairs(node,deep=deep);result={}
    for key,value in pairs:
        require(key not in result,'Duplicate dataset YAML key');result[key]=value
    return result


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,unique_mapping)


def file_hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(65536),b''):result.update(data)
    return result.hexdigest()


def labels_from_bytes(data):
    require(len(data)<=1024*1024,'Oversized annotation file')
    rows=[]
    for index,line in enumerate(data.decode('utf-8-sig').splitlines(),1):
        if not line.strip():continue
        fields=line.split();require(len(fields)==5,'Detection annotation must have five fields at line '+str(index))
        require(re.fullmatch('[0-3]',fields[0]) is not None,'Class ID outside exact four-state mapping')
        values=[float(v) for v in fields[1:]]
        require(all(math.isfinite(v) and 0<=v<=1 for v in values) and values[2]>0 and values[3]>0,'Invalid normalized box')
        rows.append([int(fields[0]),*values])
    return sorted(rows)


def source_key(name):
    # A naming hint only: this is not proof of the same subject or capture session.
    match=re.fullmatch(r'(.+)\.rf\.[0-9a-fA-F]{32}',Path(name).stem)
    return match.group(1).casefold() if match else None


def dataset_layout(config,config_bytes=None):
    config=Path(config).resolve();require(config.stat().st_size<=65536,'Dataset YAML exceeds 64KiB')
    if config_bytes is None:config_bytes=config.read_bytes()
    require(len(config_bytes)<=65536,'Dataset YAML exceeds 64KiB')
    data=yaml.load(config_bytes.decode('utf-8-sig'),Loader=UniqueSafeLoader)
    require(isinstance(data,dict),'Dataset YAML must be a mapping')
    names=data.get('names')
    if isinstance(names,dict):
        require(set(names)==set(range(4)) and all(type(k) is int for k in names),'Exact class indexes required')
        names=[names[i] for i in range(4)]
    require(names==NAMES and type(data.get('nc')) is int and data['nc']==4,'Expected exact Cry/Happy/Neutral/Sleep order')
    root=config.parent
    if 'path' in data:
        require(isinstance(data['path'],str),'Only a local directory root is supported')
        require((root/data['path']).resolve()==root,'Audit uses the explicitly supplied YAML directory as root')
    layout={}
    for split in ('train','val','test'):
        require(type(data.get(split)) is str and data[split],'This audit supports one local image directory per split')
        path=(root/data[split]).resolve()
        require(path.is_relative_to(root) and path.is_dir() and path.name=='images','Split must be a local images directory within the dataset root')
        require((path.parent/'labels').is_dir(),'Matching local labels directory required')
        layout[split]=path
    require(len(set(layout.values()))==3,'Splits cannot resolve to the same directory')
    return root,layout


def require_local_asset(path,root):
    require(path.resolve().is_relative_to(root),'Dataset asset escapes its root')
    for part in (path,*path.parents):
        if part==root:break
        require(not part.is_symlink() and not part.is_junction(),'Linked dataset assets are not supported')


def audit_dataset(config):
    config=Path(config).resolve();require(config.stat().st_size<=65536,'Dataset YAML exceeds 64KiB')
    config_bytes=config.read_bytes();root,layout=dataset_layout(config,config_bytes)
    inputs={config: hashlib.sha256(config_bytes).hexdigest()};records=[];errors=[];warnings=[];stats={};inventory=[]
    byte_groups=defaultdict(list);pixel_groups=defaultdict(list);origin_groups=defaultdict(list)
    for split,images in layout.items():
        labels=images.parent/'labels';used_labels=set();classes=Counter();background=missing=0;image_stems=set()
        image_files=sorted(p for p in images.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        label_files=sorted(p for p in labels.rglob('*.txt') if p.is_file());inventory += image_files+label_files
        require(image_files,'Every split must have at least one supported image')
        for path in image_files:
            require_local_asset(path,root)
            relative=path.relative_to(root).as_posix();raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();inputs[path]=sha
            row=dict(split=split,image=relative,imageSha256=sha,pixelSha256=None,boxes=[],label=None,labelSha256=None)
            stem=path.relative_to(images).with_suffix('').as_posix().casefold()
            if stem in image_stems:errors.append(dict(image=relative,kind='ambiguous_image_stem'))
            image_stems.add(stem)
            try:
                with Image.open(io.BytesIO(raw)) as picture:
                    if picture.getexif().get(274,1)!=1:warnings.append(dict(image=relative,kind='EXIF_orientation_requires_label_review'))
                    oriented=ImageOps.exif_transpose(picture).convert('RGB');width,height=oriented.size
                    row.update(width=width,height=height,pixelSha256=hashlib.sha256(struct.pack('>II',width,height)+oriented.tobytes()).hexdigest())
            except (OSError,ValueError,Image.DecompressionBombError):
                errors.append(dict(image=relative,kind='image_decode_failed'))
            label=labels/path.relative_to(images).with_suffix('.txt')
            if label.exists():
                require_local_asset(label,root)
                used_labels.add(label);data=label.read_bytes();label_sha=hashlib.sha256(data).hexdigest();inputs[label]=label_sha
                row.update(label=label.relative_to(root).as_posix(),labelSha256=label_sha)
                try:
                    row['boxes']=labels_from_bytes(data)
                    if len({tuple(box) for box in row['boxes']})!=len(row['boxes']):
                        warnings.append(dict(image=relative,kind='duplicate_annotation_rows'))
                except (ValueError,UnicodeError,OverflowError) as exc:
                    errors.append(dict(image=relative,kind='invalid_detection_annotation',detail=str(exc)))
            else:missing+=1
            if not row['boxes']:background+=1
            for box_index,(class_id,x,y,w,h) in enumerate(row['boxes']):
                classes[NAMES[class_id]]+=1
                if x-w/2< -1e-6 or y-h/2< -1e-6 or x+w/2>1+1e-6 or y+h/2>1+1e-6:
                    overflow=[max(0,w/2-x),max(0,h/2-y),max(0,x+w/2-1),max(0,y+h/2-1)]
                    pixels=None
                    if 'width' in row:
                        pixels=[v*scale for v,scale in zip(overflow,[row['width'],row['height']]*2)]
                    warnings.append(dict(image=relative,kind='box_corners_extend_outside_image',boxIndex=box_index,
                                         overflowLeftTopRightBottom=overflow,
                                         maxOverflowPixels=max(pixels) if pixels is not None else None))
            records.append(row);byte_groups[sha].append(row)
            if row['pixelSha256']:pixel_groups[row['pixelSha256']].append(row)
            key=source_key(path.name)
            if key:origin_groups[key].append(row)
        for orphan in set(label_files)-used_labels:
            require_local_asset(orphan,root)
            inputs[orphan]=file_hash(orphan);errors.append(dict(label=orphan.relative_to(root).as_posix(),kind='orphan_annotation'))
        stats[split]=dict(images=len(image_files),instances=sum(classes.values()),classes={name:classes[name] for name in NAMES},
                          backgroundOrUnusableImages=background,missingLabelsAllowedAsBackground=missing)
    def groups(mapping):
        return [dict(key=key,images=[q['image'] for q in rows],splits=sorted({q['split'] for q in rows}),
                     annotationConflict=len({json.dumps(q['boxes'],separators=(',',':')) for q in rows})>1)
                for key,rows in sorted(mapping.items()) if len(rows)>1]
    byte_duplicates=groups(byte_groups);pixel_duplicates=groups(pixel_groups);hints=groups(origin_groups)
    byte_cross=[g for g in byte_duplicates if len(g['splits'])>1];pixel_cross=[g for g in pixel_duplicates if len(g['splits'])>1]
    hint_cross=[g for g in hints if len(g['splits'])>1]
    # Rehash and rescan; caches are never created and no input is normalized on disk.
    for path,sha in inputs.items():require(file_hash(path)==sha,'Dataset changed while auditing')
    after=[]
    for images in layout.values():
        after += [p for p in images.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        after += [p for p in (images.parent/'labels').rglob('*.txt') if p.is_file()]
    require(set(after)==set(inventory),'Dataset file membership changed while auditing')
    return dict(schemaVersion=1,scope='local_four_state_detection_dataset_audit',root=str(root),configSha256=inputs[config],
                auditToolSha256=file_hash(Path(__file__)),
                labelFormatReference='https://docs.ultralytics.com/datasets/detect',names=NAMES,splits=stats,records=records,
                inputFiles={str(p):sha for p,sha in sorted(inputs.items())},errors=errors,warnings=warnings,
                exactByteDuplicateGroups=byte_duplicates,orientedRGBPixelDuplicateGroups=pixel_duplicates,
                crossSplitByteGroups=len(byte_cross),crossSplitPixelGroups=len(pixel_cross),
                crossSplitFilenameHintGroups=hint_cross,filenameHintsAreNotVerifiedSourceOrSubjectIDs=True,
                labelsAndDecodeValid=not errors,allSplitsHaveAllClasses=all(all(v['classes'].values()) for v in stats.values()),
                annotationReviewRequired=bool(warnings),
                noCrossSplitExactDuplicatesFound=not byte_cross and not pixel_cross,
                heldOutSubjectOrCaptureSessionIndependenceVerified=False,nearDuplicateSearchPerformed=False,
                modelInferencePerformed=False,networkAccessPerformed=False,hardwareAccessPerformed=False,
                datasetModified=False,liveCameraOrClinicalAccuracyQualified=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path,help='new local directory outside the dataset')
    args=parser.parse_args(argv)
    try:
        require(not args.output.exists(),'Existing output is never overwritten')
        require(args.data.resolve().parent not in args.output.resolve().parents,'Keep reports outside the source dataset')
        result=audit_dataset(args.data);args.output.mkdir(parents=True,exist_ok=False)
        records=result.pop('records')
        with (args.output/'images.jsonl').open('x',encoding='utf-8') as f:
            for row in records:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        with (args.output/'audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    except (OSError,ValueError,TypeError,KeyError,yaml.YAMLError,RecursionError):
        print('Local dataset audit failed; check the explicit directory configuration and new output path. No images were uploaded.');return 2
    print(json.dumps(dict(splits=result['splits'],annotationOrDecodeErrors=len(result['errors']),warnings=len(result['warnings']),
          annotationReviewRequired=result['annotationReviewRequired'],
          crossSplitByteGroups=result['crossSplitByteGroups'],crossSplitPixelGroups=result['crossSplitPixelGroups'],
          crossSplitFilenameHintGroups=len(result['crossSplitFilenameHintGroups']),subjectIndependenceVerified=False),ensure_ascii=False))
    return 0 if result['labelsAndDecodeValid'] and result['noCrossSplitExactDuplicatesFound'] else 1


if __name__=='__main__':raise SystemExit(main())
