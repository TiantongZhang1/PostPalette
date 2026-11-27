"""Prepare labeled HF HAR data; its placeholder-labeled test split is never used."""
from hashlib import sha256
from io import BytesIO
import json
import random
from pathlib import Path
from urllib.request import urlopen, Request
import numpy as np
from PIL import Image, ImageOps
from .config import DATA
from .daily_activity import CLASSES, SOURCE_LABELS, SIZE, preprocess

REPO = 'Bingsu/Human_Action_Recognition'


def dhash(image):
    pixels = np.array(image.convert('L').resize((9,8), Image.Resampling.LANCZOS))
    value = 0
    for bit in (pixels[:,1:] > pixels[:,:-1]).flat:
        value = (value<<1)|int(bit)
    return value


class SimilarityIndex:
    """Five disjoint hash bands retrieve every pair at Hamming distance <= 4."""
    def __init__(self):
        self.bands = {}; self.values = []

    @staticmethod
    def keys(value):
        return [(i,(value>>shift)&((1<<width)-1)) for i,(shift,width) in enumerate(((0,13),(13,13),(26,13),(39,13),(52,12)))]

    def matches(self,values):
        candidates = set()
        for value in values:
            for key in self.keys(value):candidates.update(self.bands.get(key, ()))
        return [i for i in candidates if any((a^b).bit_count()<=4 for a in values for b in self.values[i])]

    def add(self,values):
        index = len(self.values); self.values.append(values)
        for value in values:
            for key in self.keys(value):self.bands.setdefault(key,set()).add(index)


def download(url,path):
    if path.exists():return
    temp = path.with_suffix('.partial')
    with urlopen(Request(url,headers={'User-Agent':'PostPalette-course-data/1.0'}),timeout=90) as response, temp.open('wb') as output:
        total = 0
        while chunk:=response.read(1024*1024):
            output.write(chunk);total+=len(chunk)
            if total%(32*1024*1024)==0:print(f'Downloaded {total//(1024*1024)} MiB',flush=True)
    temp.replace(path)


def main():
    import pyarrow.parquet as pq
    directory = DATA/'daily_activity'; directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():
        print('Activity data already prepared; keeping the existing split.',flush=True);return
    revision_path = directory/'source_revision.json'
    if not revision_path.exists():
        with urlopen(f'https://huggingface.co/api/datasets/{REPO}',timeout=45) as response:metadata=json.load(response)
        revision_path.write_text(json.dumps({'repository':REPO,'revision':metadata['sha']}),encoding='utf-8')
    revision = json.loads(revision_path.read_text())['revision']
    path = directory/'labeled_train.parquet'
    download(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/data/train-00000-of-00001.parquet',path)
    index = SimilarityIndex(); records = []; parent = []
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            label = int(row['labels'])
            if not 0<=label<len(SOURCE_LABELS):raise ValueError('Unknown source label')
            content = row['image']['bytes']
            with Image.open(BytesIO(content)) as original:
                rgb = ImageOps.exif_transpose(original).convert('RGB')
                hashes = (dhash(rgb),dhash(ImageOps.mirror(rgb)))
                matches = index.matches(hashes);n=len(records);parent.append(n)
                for match in matches:parent[find(match)]=find(n)
                records.append({'source_index':n,'source_label':SOURCE_LABELS[label],
                                'image':preprocess(rgb),'area':rgb.width*rgb.height,
                                'sha256':sha256(rgb.tobytes()).hexdigest(),'hashes':hashes})
                index.add(hashes)
        if len(records)%1024==0:print(f'Decoded/grouped {len(records)} labeled photos',flush=True)
    groups = {}
    for i,row in enumerate(records):groups.setdefault(find(i),[]).append(row)
    representatives=[];conflicting=0
    for rows in groups.values():
        if len({row['source_label'] for row in rows})>1:conflicting+=len(rows);continue
        representative=max(rows,key=lambda r:r['area'])
        representative['group_members']=[r['source_index'] for r in rows]; representatives.append(representative)
    splits={s:[] for s in ('train','val','test')};rng=random.Random(2026)
    for category in SOURCE_LABELS:
        rows=sorted((r for r in representatives if r['source_label']==category),key=lambda r:r['source_index'])
        rng.shuffle(rows);v=max(1,round(len(rows)*.1));t=max(1,round(len(rows)*.1))
        for split,subset in (('val',rows[:v]),('test',rows[v:v+t]),('train',rows[v+t:])):
            for row in subset:
                row['label']=category if category in CLASSES else 'other';row['source']='HF labeled train'
                splits[split].append(row)
    # Add a small independent sample of no-person photos to help reject unrelated uploads.
    coco_counts={}
    for split,limit in (('train',800),('val',160),('test',160)):
        metadata_path=DATA/'people'/f'{split}.json';coco_counts[split]=0
        if not metadata_path.exists():continue
        rows=[r for r in json.loads(metadata_path.read_text()) if not r['people'] and not r['crowds']]
        rng.shuffle(rows)
        for row in rows:
            with Image.open(DATA/'people/images'/f"{row['id']}-192.jpg") as image:
                hashes=(dhash(image),dhash(ImageOps.mirror(image)))
                if index.matches(hashes):continue
                index.add(hashes)
                splits[split].append({'source':'COCO no-person','source_index':row['id'],'source_label':'no_person',
                                      'group_members':[],'label':'other','image':preprocess(image)})
                coco_counts[split]+=1
                if coco_counts[split]>=limit:break
    counts={}
    for split,rows in splits.items():
        rng.shuffle(rows)
        np.savez(directory/f'{split}.npz',images=np.stack([r['image'] for r in rows]),
                 labels=np.array([CLASSES.index(r['label']) for r in rows],dtype=np.int64))
        metadata=[{k:v for k,v in r.items() if k not in ('image','hashes','area')} for r in rows]
        (directory/f'{split}_sources.json').write_text(json.dumps(metadata),encoding='utf-8')
        counts[split]={'images':len(rows),'per_class':{label:sum(r['label']==label for r in rows) for label in CLASSES}}
    manifest={'source':f'https://huggingface.co/datasets/{REPO}','revision':revision,
              'parquet_sha256':sha256(path.read_bytes()).hexdigest(),'source_labels':SOURCE_LABELS,'labels':CLASSES,'size':SIZE,
              'seed':2026,'splits':counts,'original_labeled_images':len(records),'conflicting_group_images_dropped':conflicting,
              'deduplication':'dHash64 distance <=4 including mirrored images; connected groups; one highest-resolution representative; conflicting source-label groups discarded',
              'split_rule':'stratified 80/10/10 by original activity label, after deduplication; COCO negatives keep their existing train/val/test boundaries',
              'coco_no_person_counts':coco_counts,'unused_source_test':'HF test has placeholder zero labels; not downloaded or evaluated',
              'limitations':'No subject or original web-source IDs. Hash deduplication does not guarantee identity/source independence. One dominant activity label per photo; not per-person or multi-action ground truth.'}
    temp=directory/'manifest.partial';temp.write_text(json.dumps(manifest,indent=2),encoding='utf-8');temp.replace(directory/'manifest.json')
    print(json.dumps({'splits':counts,'conflicts':conflicting}),flush=True)


if __name__=='__main__':main()
