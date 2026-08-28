#!/usr/bin/env python3
from pathlib import Path
import hashlib, argparse, subprocess, sys

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('runtime/rebuilt_train_fingerprints.csv')); a=ap.parse_args()
    r=a.data_root; a.out.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(r/'build_fingerprints.py'),'-i',str(r/'BLE_processed.csv'),'-l',str(r/'user_97.csv'),'-o',str(a.out),'--window','10','--step','2','--features','mean,present,std,max,count','--past-context','8','--past-feats','count,std']
    subprocess.run(cmd,check=True)
    expected=r/'train_fingerprints.csv'; print('rebuilt ',sha(a.out)); print('expected',sha(expected)); print('match',sha(a.out)==sha(expected))
if __name__=='__main__': main()
