"""Synthetic regression tests for new reconstruction; no study results generated."""
import csv
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import reconstruct as r
from metrics import select_thresholds
from data import load_archive
from model import CompactCNN
from train import train_neural


class ReconstructionTests(unittest.TestCase):
    def test_fast_thresholds_equal_brute_force(self):
        rng=np.random.default_rng(3);y=rng.integers(0,2,(35,3));p=rng.random((35,3))
        p[0]=p[1];y[:,2]=0
        expected=[]
        for j in range(3):
            vals=[(f1_score(y[:,j],p[:,j]>=t,zero_division=0),t) for t in np.unique(np.round(p[:,j],6))]
            expected.append(max(vals)[1])
        np.testing.assert_array_equal(select_thresholds(y,p),expected)

    def test_frozen_evaluation_and_integrity(self):
        rng=np.random.default_rng(8);counts={'train':40,'val':24,'test':32}
        previous=dict(r.SPLITS);r.SPLITS.clear();r.SPLITS.update(counts)
        try:
            with tempfile.TemporaryDirectory() as d:
                out=Path(d);arc=out/'synthetic.npz';mapping=out/'mapping.csv'
                arrays={}
                for s,n in counts.items():
                    arrays[s+'_images']=rng.integers(0,256,(n,64,64),dtype=np.uint8)
                    arrays[s+'_labels']=rng.integers(0,2,(n,14),dtype=np.uint8)
                arrays['test_images'][0]=arrays['train_images'][0]
                np.savez(arc,**arrays)
                with mapping.open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['split','index','source_filename'])
                    for s,n in counts.items():w.writerows((s,i,f'{s}{i//2:04d}_000.png') for i in range(n))
                loader=lambda archive,splits=None:load_archive(archive,verify=False,splits=splits)
                with patch.object(r,'load_archive',loader):
                    r.integrity_stage(arc,mapping,out)
                    np.save(out/'B0_prevalence.npy',arrays['train_labels'].mean(0))
                    np.save(out/'B0_val_probs.npy',np.tile(arrays['train_labels'].mean(0),(counts['val'],1)))
                    np.save(out/'B1_val_probs.npy',rng.random((counts['val'],14)))
                    torch.save(torch.nn.Linear(4096,14).state_dict(),out/'B1_wd0.001.pt')
                    r.save_json(out/'B1_selection.json',{'selected_weight_decay':0.001})
                    for condition in ['B2','B3']:
                        for seed in r.SEEDS:
                            name=f'{condition}_seed{seed}'
                            torch.save(CompactCNN().state_dict(),out/f'{name}.pt')
                            np.save(out/f'{name}_val_probs.npy',rng.random((counts['val'],14)))
                            r.save_json(out/f'{name}_complete.json',{'synthetic_fixture':True})
                            r.save_json(out/f'{name}_history.json',{'synthetic_fixture':True})
                    r.freeze_stage(arc,mapping,out)
                    r.evaluation_stage(arc,mapping,out,bootstrap_reps=8)
                    summary=json.loads((out/'locked_test_summary.json').read_text())
                    self.assertEqual(summary['duplicate_sensitivity']['excluded_test_indices'],[0])
                    self.assertIn('B1',summary);self.assertIn('B0',summary)
                    self.assertEqual(np.load(out/'bootstrap_replicates.npy').shape,(8,))
                    # Reanalysis uses saved predictions and must not reopen raw test data.
                    with patch.object(r,'load_archive',side_effect=RuntimeError('raw data reopened')):
                        r.evaluation_stage(arc,mapping,out,bootstrap_reps=8)
                    (out/'thresholds.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError,'Frozen artifact changed'):
                        r.evaluation_stage(arc,mapping,out,bootstrap_reps=8)
        finally:r.SPLITS.clear();r.SPLITS.update(previous)

    def test_resume_preserves_training_result(self):
        rng=np.random.default_rng(2)
        data={s:(rng.integers(0,256,(n,64,64),dtype=np.uint8),rng.integers(0,2,(n,14))) for s,n in [('train',20),('val',20)]}
        with tempfile.TemporaryDirectory() as d:
            complete=Path(d)/'complete';resumed=Path(d)/'resumed'
            train_neural('B2',20261,data,str(complete))
            real_save=torch.save
            def interrupted_save(obj,path,*args,**kwargs):
                real_save(obj,path,*args,**kwargs)
                if str(path).endswith('_resume.pt.tmp') and obj['epoch']==2:
                    os.replace(path,str(path)[:-4]);raise RuntimeError('simulated interruption')
            with patch('train.torch.save',side_effect=interrupted_save):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    train_neural('B2',20261,data,str(resumed))
            train_neural('B2',20261,data,str(resumed))
            a=torch.load(complete/'B2_seed20261.pt',weights_only=True)
            b=torch.load(resumed/'B2_seed20261.pt',weights_only=True)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
