"""Audit, train, freeze and evaluate a new ChestMNIST reconstruction.

The original experiment archive is unavailable. This entry point creates new
results and does not certify agreement with the manuscript's reported numbers.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from importlib.metadata import version

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, hamming_loss, accuracy_score)
from data import load_archive, audit, SPLITS, train_statistics, standardise
from model import CompactCNN
from train import train_neural, train_linear, predict
from metrics import (macro_ap, macro_auc, macro_brier, per_label_ap, per_label_auc,
                     select_thresholds, paired_cluster_bootstrap)

SEEDS = [20261, 20262, 20263]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def mapping_records(path):
    with open(path) as f: rows = list(csv.DictReader(f))
    by_split = {s: {} for s in SPLITS}
    for row in rows:
        split = row['split']; index = int(row['index'])
        if split not in SPLITS or not 0 <= index < SPLITS[split]:
            raise ValueError('Invalid source-mapping split/index')
        if index in by_split[split]: raise ValueError('Duplicate source-mapping row')
        name = row['source_filename']
        if not name or '_' not in name: raise ValueError('Missing source filename/patient prefix')
        by_split[split][index] = name
    for split in SPLITS:
        if set(by_split[split]) != set(range(SPLITS[split])):
            raise ValueError(f'Incomplete source mapping: {split}')
    return by_split


def integrity_stage(archive, mapping, out):
    data = load_archive(archive)
    report = audit(data, str(out / 'data_audit.json'))
    records = mapping_records(mapping)
    prefixes = {s: {n.split('_')[0] for n in v.values()} for s, v in records.items()}
    crossing = set.union(*(prefixes[a] & prefixes[b] for a, b in [('train','val'),('train','test'),('val','test')]))
    save_json(out / 'source_mapping_audit.json', {
        'mapping_sha256': sha(mapping), 'rows': {s: len(v) for s,v in records.items()},
        'patient_prefixes': {s: len(v) for s,v in prefixes.items()},
        'cross_split_patient_prefix_count': len(crossing),
        'cross_split_patient_prefixes': sorted(crossing),
        'source': 'official MedMNIST source mapping; image_id normalised to source_filename'
    })
    excluded = sorted({int(index) for members in report['cross_split_detail'].values()
                       for split, index, _ in members if split == 'test'})
    save_json(out / 'duplicate_test_exclusions.json', excluded)
    mean, std = train_statistics(data['train'][0])
    save_json(out / 'preprocessing.json', {'mean': mean, 'std': std, 'source_split': 'train'})
    save_json(out / 'new_run.json', {
        'run_kind': 'new_reconstruction', 'original_artifacts_available': False,
        'archive_sha256': sha(archive), 'archive': Path(archive).name,
        'mapping_sha256': sha(mapping), 'python': platform.python_version(),
        'packages': {p: version(p) for p in ['numpy','scipy','scikit-learn','torch']},
        'threads': torch.get_num_threads(), 'seeds': SEEDS,
        'max_epochs': 20, 'early_stop_patience': 4,
        'precision': 'float32', 'memory_format': 'channels_last',
        'audit_note': 'Test arrays are inspected for data integrity only before training; performance evaluation follows model freezing.'
    })
    print('Real-data integrity audit completed.', flush=True)


def training_stage(archive, out):
    provenance = json.loads((out / 'new_run.json').read_text())
    if sha(archive) != provenance['archive_sha256']:
        raise ValueError('Data archive changed since the integrity audit')
    save_json(out / 'training_environment.json', {
        'threads': torch.get_num_threads(), 'python': platform.python_version(),
        'torch': version('torch'), 'precision': 'float32', 'memory_format': 'channels_last'})
    data = load_archive(archive, splits=['train','val'])
    prevalence = data['train'][1].mean(axis=0)
    np.save(out / 'B0_prevalence.npy', prevalence)
    np.save(out / 'B0_val_probs.npy', np.tile(prevalence, (len(data['val'][1]), 1)))
    if not (out / 'B1_selection.json').exists(): train_linear(data, str(out))
    for seed in SEEDS:
        for condition in ['B2','B3']:
            train_neural(condition, seed, data, str(out))


def freeze_stage(archive, mapping, out):
    if (out / 'test_access.json').exists(): raise ValueError('Cannot refreeze after test evaluation starts')
    val = load_archive(archive, splits=['val'])['val'][1]
    predictions = {name: np.load(out / f'{name}_val_probs.npy') for name in ['B0','B1']}
    artifacts = ['new_run.json','preprocessing.json','B0_prevalence.npy','B1_selection.json']
    selected = json.loads((out / 'B1_selection.json').read_text())['selected_weight_decay']
    artifacts.append(f'B1_wd{selected:g}.pt')
    for condition in ['B2','B3']:
        parts = []
        for seed in SEEDS:
            name = f'{condition}_seed{seed}'
            if not (out / f'{name}_complete.json').exists(): raise ValueError(f'Training incomplete: {name}')
            predictions[name] = np.load(out / f'{name}_val_probs.npy')
            parts.append(predictions[name]); artifacts.extend([f'{name}.pt',f'{name}_history.json'])
        predictions[condition] = np.mean(parts, axis=0)
    thresholds = {name: select_thresholds(val, p) for name,p in predictions.items()}
    save_json(out / 'thresholds.json', thresholds); artifacts.append('thresholds.json')
    save_json(out / 'freeze.json', {
        'archive_sha256': sha(archive), 'mapping_sha256': sha(mapping),
        'protocol_sha256': sha('configs/protocol.yaml'),
        'artifacts': {name: sha(out / name) for name in artifacts},
        'selection_data': 'validation only', 'test_access_started': False,
        'run_kind': 'new_reconstruction'
    })
    print('Models and validation thresholds frozen.', flush=True)


def measures(y, p, thresholds):
    binary = p >= np.asarray(thresholds)
    negative = y == 0
    specificity = ((~binary) & negative).sum(0) / np.maximum(negative.sum(0), 1)
    return {'macro_ap': macro_ap(y,p), 'micro_ap': float(average_precision_score(y,p,average='micro')),
            'macro_auc': macro_auc(y,p), 'micro_auc': float(roc_auc_score(y,p,average='micro')),
            'macro_brier': macro_brier(y,p), 'macro_f1': float(f1_score(y,binary,average='macro',zero_division=0)),
            'micro_f1': float(f1_score(y,binary,average='micro',zero_division=0)),
            'macro_precision': float(precision_score(y,binary,average='macro',zero_division=0)),
            'macro_recall': float(recall_score(y,binary,average='macro',zero_division=0)),
            'macro_specificity': float(specificity.mean()),
            'hamming_loss': float(hamming_loss(y,binary)), 'exact_match_accuracy': float(accuracy_score(y,binary)),
            'per_label_ap': per_label_ap(y,p), 'per_label_auc': per_label_auc(y,p),
            'per_label_brier': ((p-y)**2).mean(0).tolist()}


def evaluation_stage(archive, mapping, out, bootstrap_reps=2000):
    frozen = json.loads((out / 'freeze.json').read_text())
    for name, expected in frozen['artifacts'].items():
        if sha(out/name) != expected: raise ValueError(f'Frozen artifact changed: {name}')
    for path,key in [(archive,'archive_sha256'),(mapping,'mapping_sha256'),('configs/protocol.yaml','protocol_sha256')]:
        if sha(path) != frozen[key]: raise ValueError(f'Frozen input changed: {key}')
    cached = out / 'locked_test_predictions.npz'
    if cached.exists():
        arrays = np.load(cached,allow_pickle=False); y = arrays['labels']
        predictions = {k: arrays[k] for k in arrays.files if k != 'labels'}
    else:
        if (out / 'test_access.json').exists(): raise ValueError('Previous test prediction stage was interrupted; no complete prediction archive is available')
        save_json(out / 'test_access.json', {'started': True, 'new_run': True})
        x,y = load_archive(archive,splits=['test'])['test']
        pre = json.loads((out / 'preprocessing.json').read_text())
        xt = torch.tensor(standardise(x,pre['mean'],pre['std'])).unsqueeze(1)
        predictions = {'B0': np.tile(np.load(out / 'B0_prevalence.npy'),(len(y),1))}
        wd=json.loads((out / 'B1_selection.json').read_text())['selected_weight_decay']
        linear=torch.nn.Linear(4096,14)
        linear.load_state_dict(torch.load(out/f'B1_wd{wd:g}.pt',weights_only=True,map_location='cpu'))
        predictions['B1']=predict(linear,xt.reshape(len(y),-1))
        for condition in ['B2','B3']:
            ps=[]
            for seed in SEEDS:
                model=CompactCNN().to(memory_format=torch.channels_last)
                model.load_state_dict(torch.load(out/f'{condition}_seed{seed}.pt',weights_only=True,map_location='cpu'))
                name=f'{condition}_seed{seed}';predictions[name]=predict(model,xt);ps.append(predictions[name])
            predictions[condition]=np.mean(ps,axis=0)
        np.savez_compressed(cached,labels=y,**predictions)
    thresholds=json.loads((out/'thresholds.json').read_text())
    summary={name:measures(y,p,thresholds[name]) for name,p in predictions.items()}
    records=mapping_records(mapping)['test'];clusters=np.array([records[i].split('_')[0] for i in range(len(y))])
    with (out/'test_records.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','source_filename','patient_prefix'])
        w.writerows((i,records[i],clusters[i]) for i in range(len(y)))
    boot=paired_cluster_bootstrap(y,predictions['B2'],predictions['B3'],clusters,n_rep=bootstrap_reps)
    np.save(out/'bootstrap_replicates.npy',np.array(boot.pop('replicates')))
    summary['primary_contrast_B3_minus_B2']=boot
    excluded=json.loads((out/'duplicate_test_exclusions.json').read_text())
    keep=np.ones(len(y),dtype=bool);keep[excluded]=False
    summary['duplicate_sensitivity']={'excluded_test_indices':excluded,'remaining_rows':int(keep.sum()),
        'B2':measures(y[keep],predictions['B2'][keep],thresholds['B2']),
        'B3':measures(y[keep],predictions['B3'][keep],thresholds['B3'])}
    d=np.array(summary['B2']['per_label_ap'])-np.array(summary['B3']['per_label_ap']);nonzero=d[d!=0]
    summary['corroboration']={'labels_favoring_B2':int((d>0).sum()),
        'sign_test_p':float(stats.binomtest(int((nonzero>0).sum()),len(nonzero),0.5).pvalue) if len(nonzero) else 1.0,
        'wilcoxon_ap_p':float(stats.wilcoxon(d).pvalue) if len(nonzero) else 1.0}
    summary['provenance']={'new_reconstruction':True,'original_result_agreement_not_assumed':True,'bootstrap_replicates':bootstrap_reps}
    save_json(out/'locked_test_summary.json',summary)
    save_json(out/'evaluation_complete.json',{'complete':True,'prediction_sha256':sha(cached)})
    print(json.dumps(summary['primary_contrast_B3_minus_B2']),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['audit','train','freeze','evaluate','all'],default='all')
    ap.add_argument('--archive',default='chestmnist_64.npz');ap.add_argument('--mapping',default='source_mapping.csv')
    ap.add_argument('--out',default='outputs');ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args();torch.set_num_threads(a.threads);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    stages=['audit','train','freeze','evaluate'] if a.stage=='all' else [a.stage]
    for stage in stages:
        print('STAGE',stage,flush=True)
        if stage=='audit': integrity_stage(a.archive,a.mapping,out)
        elif stage=='train': training_stage(a.archive,out)
        elif stage=='freeze': freeze_stage(a.archive,a.mapping,out)
        else:evaluation_stage(a.archive,a.mapping,out)


if __name__=='__main__':main()
