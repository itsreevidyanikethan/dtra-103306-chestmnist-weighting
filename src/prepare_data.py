"""Download the official archive and normalise its separately obtained row mapping."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

URL='https://zenodo.org/records/10519652/files/chestmnist_64.npz?download=1'
MD5='9de6cd0b934ebb5b7426cfba5efbae16'
MAPPING_URL='https://drive.google.com/file/d/1eY4En_i6XYWzwzXxI2JKDsFspW-ho__I/view'


def digest(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def prepare(archive, original, output):
    archive=Path(archive);original=Path(original);output=Path(output)
    if not original.is_file():raise FileNotFoundError('Download the official mapping first: '+MAPPING_URL)
    if not archive.exists():
        temporary=archive.with_suffix('.npz.part')
        with urlopen(URL, timeout=60) as source, temporary.open('wb') as target:
            while chunk:=source.read(8*1024*1024):target.write(chunk)
        if digest(temporary,'md5')!=MD5:raise ValueError('Official archive MD5 mismatch')
        temporary.replace(archive)
    if digest(archive,'md5')!=MD5:raise ValueError('Official archive MD5 mismatch')
    if original.resolve()==output.resolve():raise ValueError('Keep the original mapping separate')
    with original.open(newline='') as source:
        reader=csv.DictReader(source)
        if not {'split','index','image_id'}<=set(reader.fieldnames or []):raise ValueError('Unexpected mapping columns')
        rows=list(reader)
    seen=set();counts={'train':78468,'val':11219,'test':22433}
    converted=[]
    for row in rows:
        split=row['split'];index=int(row['index']);name=row['image_id']
        key=(split,index)
        if split not in counts or not 0<=index<counts[split] or key in seen or '_' not in name:
            raise ValueError('Invalid mapping row')
        seen.add(key);converted.append([split,index,name if name.endswith('.png') else name+'.png'])
    if len(seen)!=sum(counts.values()):raise ValueError('Incomplete mapping')
    with output.open('w',newline='') as target:
        writer=csv.writer(target);writer.writerow(['split','index','source_filename']);writer.writerows(converted)
    report={'archive_url':URL,'archive_md5':MD5,'archive_sha256':digest(archive),
            'mapping_url':MAPPING_URL,'original_mapping_sha256':digest(original),
            'normalised_mapping_sha256':digest(output),'rows':len(rows)}
    Path('data_source.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Archive verified and mapping normalised:',len(rows),'rows')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',default='chestmnist_64.npz');ap.add_argument('--mapping-original',required=True)
    ap.add_argument('--mapping-out',default='source_mapping.csv');a=ap.parse_args()
    prepare(a.archive,a.mapping_original,a.mapping_out)
