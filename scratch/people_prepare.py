"""Prepare higher-resolution people/pose, action and expression datasets without pretrained processing."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageOps

from .config import DATA


def download(url, path):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    with urlopen(Request(url, headers={'User-Agent':'PostPalette-research/1.0'}), timeout=60) as response, temporary.open('wb') as out:
        while block := response.read(1024 * 1024):
            out.write(block)
    temporary.replace(path)
    return path


def json_save(path, content):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def pixels(content, size, gray=False):
    with Image.open(BytesIO(content)) as image:
        image = ImageOps.exif_transpose(image).convert('L' if gray else 'RGB')
        image = ImageOps.pad(image, (size,size), Image.Resampling.LANCZOS, color=127 if gray else (127,127,127))
        array = np.array(image)
    return array[None] if gray else array.transpose(2,0,1)


def people(args):
    directory = DATA/'people'
    directory.mkdir(exist_ok=True)
    rng = random.Random(2026)
    metadata = {}
    source_hash = {}
    with ZipFile(DATA/'coco/annotations_trainval2017.zip') as archive:
        for split in ('train2017','val2017'):
            raw = archive.read(f'annotations/person_keypoints_{split}.json')
            source_hash[split] = sha256(raw).hexdigest()
            document = json.loads(raw)
            del raw
            rows = {r['id']: {'id':r['id'],'filename':r['file_name'],'width':r['width'],'height':r['height'],
                              'official_split':split,'people':[],'crowds':[]} for r in document['images']}
            for annotation in document['annotations']:
                row = rows[annotation['image_id']]
                if annotation.get('iscrowd'):
                    row['crowds'].append(annotation['bbox'])
                else:
                    scale=args.size/max(row['width'],row['height'])
                    if min(annotation['bbox'][2:])*scale >= 5:
                        row['people'].append({'bbox':annotation['bbox'],'keypoints':annotation.get('keypoints',[0]*51)})
            if split=='train2017':
                positive=[r for r in rows.values() if r['people']]
                negative=[r for r in rows.values() if not r['people'] and not r['crowds']]
                rng.shuffle(positive);rng.shuffle(negative)
                # Balanced count buckets prevent only learning common single-person scenes.
                buckets={i:[] for i in range(1,6)}
                for row in positive:buckets[min(len(row['people']),5)].append(row)
                selected=[]
                while len(selected)<int(args.train*0.8) and any(buckets.values()):
                    for bucket in buckets.values():
                        if bucket and len(selected)<int(args.train*0.8):selected.append(bucket.pop())
                metadata['train']=selected+negative[:args.train-len(selected)]
                rng.shuffle(metadata['train'])
            else:
                values=list(rows.values());rng.shuffle(values)
                metadata['val']=values[:args.val]
                metadata['test']=values[args.val:args.val+args.test]
            del document
    def fetch(row):
        folder=directory/'images';folder.mkdir(exist_ok=True)
        path=folder/f"{row['id']}-{args.size}.jpg"
        if path.exists():return row
        url=f"https://s3.amazonaws.com/images.cocodataset.org/{row['official_split']}/{row['filename']}"
        for attempt in range(3):
            try:
                with urlopen(url,timeout=25) as response:content=response.read(16*1024*1024)
                with Image.open(BytesIO(content)) as original:
                    if original.size != (row['width'],row['height']):raise ValueError('COCO image dimensions differ')
                    # COCO boxes use the original stored orientation; do not rotate independently.
                    image=ImageOps.pad(original.convert('RGB'),(args.size,args.size),Image.Resampling.LANCZOS,color=(127,127,127))
                    image.save(path.with_suffix('.partial'),format='JPEG',quality=95)
                path.with_suffix('.partial').replace(path)
                return row
            except (OSError,ValueError):
                if attempt==2:raise
                time.sleep(attempt+0.5)
    failures=[];valid=set();tasks=[r for rows in metadata.values() for r in rows]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fetch,row):row for row in tasks}
        for n,future in enumerate(as_completed(futures),1):
            try:valid.add(future.result()['id'])
            except Exception as error:failures.append({'id':futures[future]['id'],'error':type(error).__name__})
            if n%200==0:print(f'People photos {n}/{len(tasks)}, failures {len(failures)}',flush=True)
    for split,rows in metadata.items():
        rows=[r for r in rows if r['id'] in valid]
        json_save(directory/f'{split}.json',rows)
        array=np.lib.format.open_memmap(directory/f'{split}-{args.size}.npy',mode='w+',dtype=np.uint8,shape=(len(rows),3,args.size,args.size))
        for i,row in enumerate(rows):
            with Image.open(directory/'images'/f"{row['id']}-{args.size}.jpg") as image:array[i]=np.array(image).transpose(2,0,1)
        array.flush();del array
    json_save(directory/'manifest.json',{'source':'COCO 2017 official person keypoints','annotation_sha256':source_hash,'size':args.size,
          'seed':2026,'split_counts':{k:sum(r['id'] in valid for r in v) for k,v in metadata.items()},'failures':failures,
          'min_person_side_pixels':5,'sampling':'80% people, balanced across 1,2,3,4,5+ count buckets; 20% negatives',
          'warning':'Count individually annotated people of resolvable size; crowded regions are ignored, not counted as zero.'})


def expression(args):
    import pyarrow.parquet as pq
    directory=DATA/'expression';directory.mkdir(exist_ok=True)
    labels=['angry','disgust','fear','happy','neutral','sad','surprise']
    data={};seen=set();rng=random.Random(2026)
    for source_split in ('test','train'):
        url=f'https://huggingface.co/api/datasets/Jeneral/fer-2013/parquet/default/{source_split}/0.parquet'
        path=download(url,directory/f'{source_split}.parquet')
        rows=pq.read_table(path).to_pylist()
        converted=[]
        for row in rows:
            image=pixels(bytes(row['img_bytes']),48,True)
            digest=sha256(image.tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest);converted.append((image,int(row['labels'])))
        if source_split=='test':data['test']=converted
        else:
            rng.shuffle(converted);data['val']=converted[:2500];data['train']=converted[2500:]
    for split,rows in data.items():
        np.savez(directory/f'{split}.npz',images=np.stack([r[0] for r in rows]),labels=np.array([r[1] for r in rows]))
    json_save(directory/'manifest.json',{'source':'https://huggingface.co/datasets/Jeneral/fer-2013','source_revision':'3a46cbfae3f5b348449335f300666a0ae330f121',
             'format':'HF auto-converted parquet; no dataset script/pickle executed','labels':labels,'split_counts':{k:len(v) for k,v in data.items()},
             'deduplication':'exact decoded bitmap, test first; train held out 2500 for validation','warning':'Expression category labels, not proof of inner emotional state.',
             'files':{p.name:sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.parquet')}})
    print('Expression data ready: '+str({k:len(v) for k,v in data.items()}),flush=True)


def action(args):
    directory=DATA/'actions';directory.mkdir(exist_ok=True)
    base='https://vision.stanford.edu/Datasets/'
    images=download(base+'Stanford40.zip',directory/'Stanford40.zip')
    annotations=download(base+'Stanford40_XMLAnnotations.zip',directory/'annotations.zip')
    splits=download(base+'Stanford40_ImageSplits.zip',directory/'splits.zip')
    rng=random.Random(2026);rows={'train':[],'val':[],'test':[]}
    with ZipFile(images) as archive,ZipFile(annotations) as boxes,ZipFile(splits) as sets:
        photo_names={Path(n).name:n for n in archive.namelist() if n.lower().endswith('.jpg')}
        box_names={Path(n).name:n for n in boxes.namelist() if n.lower().endswith('.xml')}
        set_files={Path(n).name:n for n in sets.namelist()}
        train_names=sets.read(set_files['train.txt']).decode().split()
        test_names=sets.read(set_files['test.txt']).decode().split()
        classes=sorted({name.rsplit('_',1)[0] for name in photo_names})
        for label,category in enumerate(classes):
            names=[n for n in train_names if n.rsplit('_',1)[0]==category];rng.shuffle(names)
            selected={'val':names[:max(1,len(names)//5)],'train':names[max(1,len(names)//5):],
                      'test':[n for n in test_names if n.rsplit('_',1)[0]==category]}
            for split,names in selected.items():
                for name in names:
                    xml=ET.fromstring(boxes.read(box_names[name.rsplit('.',1)[0]+'.xml']))
                    box=xml.find('.//bndbox')
                    coords=[int(float(box.findtext(k))) for k in ('xmin','ymin','xmax','ymax')]
                    with Image.open(BytesIO(archive.read(photo_names[name]))) as image:
                        # XML coordinates are one-based inclusive; crop principal actor plus context.
                        x1,y1,x2,y2=coords;w=x2-x1+1;h=y2-y1+1
                        crop=image.crop((max(0,int(x1-1-.15*w)),max(0,int(y1-1-.15*h)),min(image.width,int(x2+.15*w)),min(image.height,int(y2+.15*h)))).convert('RGB')
                        crop=ImageOps.pad(crop,(128,128),Image.Resampling.LANCZOS,color=(127,127,127))
                        rows[split].append((np.array(crop).transpose(2,0,1),label,name))
    # Exact image duplicates must not cross partitions.
    seen=set()
    for split in ('test','val','train'):
        cleaned=[]
        for row in rows[split]:
            digest=sha256(row[0].tobytes()).hexdigest()
            if digest not in seen:seen.add(digest);cleaned.append(row)
        rows[split]=cleaned
        np.savez(directory/f'{split}.npz',images=np.stack([r[0] for r in cleaned]),labels=np.array([r[1] for r in cleaned]))
        json_save(directory/f'{split}_sources.json',[r[2] for r in cleaned])
    json_save(directory/'manifest.json',{'source':'https://vision.stanford.edu/Datasets/40actions.html','labels':classes,
              'preprocessing':'128px principal-actor crop with 15% context; official boxes, no pretrained detector',
              'split':'official train/test; stratified 20% of train for validation; exact crop dedup, test priority',
              'split_counts':{k:len(v) for k,v in rows.items()},'files':{p.name:sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.zip')}})
    print('Action data ready: '+str({k:len(v) for k,v in rows.items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('people','expression','action'))
    parser.add_argument('--train',type=int,default=30000)
    parser.add_argument('--val',type=int,default=1000)
    parser.add_argument('--test',type=int,default=1000)
    parser.add_argument('--size',type=int,default=192)
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.task=='people' and args.val+args.test>5000:parser.error('Validation subsets exceed official val2017')
    globals()[args.task](args)


if __name__=='__main__':main()
