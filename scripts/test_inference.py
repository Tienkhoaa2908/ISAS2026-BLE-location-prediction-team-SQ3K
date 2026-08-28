#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, hashlib, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_SHA256 = '4b1a80a214b53b531a40467a515e2afbe4350b7b0533144ec1415c3eb9a43d80'

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def hms_to_seconds(series: pd.Series) -> np.ndarray:
    p=series.astype(str).str.split(':',expand=True).astype(float)
    return (p[0]*3600+p[1]*60+p[2]).to_numpy(float)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('SQ3K_prediction.csv'))
    ap.add_argument('--workdir',type=Path,default=Path('runtime/test_inference'))
    a=ap.parse_args(); root=a.data_root.resolve(); a.workdir.mkdir(parents=True,exist_ok=True); a.out.parent.mkdir(parents=True,exist_ok=True)

    for p in [root,root/'models',root/'explore_method_resample',root/'explore_temporal_smoothing']:
        sys.path.insert(0,str(p))
    import model_core, train_resample
    common_path=Path(__file__).resolve().parent.parent/'src'/'dlcs'/'common.py'
    spec=importlib.util.spec_from_file_location('dlcs_common_submission',common_path); common=importlib.util.module_from_spec(spec); spec.loader.exec_module(common)

    train=pd.read_csv(root/'train_fingerprints.csv'); label_col=model_core.pick_label_column(train,None); feat_cols,_=model_core.select_features(train,label_col,[],0.5)
    Xtrain=train[feat_cols].apply(pd.to_numeric,errors='coerce'); y=train[label_col].astype(str); classes=sorted(y.unique()); c2i={c:i for i,c in enumerate(classes)}
    raw=pd.read_csv(root/'BLE_Test_predict.csv'); ts=pd.to_datetime(raw['timestamp'],errors='raise')
    processed=pd.DataFrame({'date':ts.dt.strftime('%Y-%m-%d'),'time':ts.dt.strftime('%H:%M:%S.%f').str.rstrip('0').str.rstrip('.'),'beacon':pd.to_numeric(raw['mac address']).astype(int),'rssi':pd.to_numeric(raw['RSSI']).astype(float)})
    days=sorted(processed['date'].unique())
    labels=pd.DataFrame({'date':days,'start_time':['00:00:00']*len(days),'finish_time':['23:59:59.999']*len(days),'room':['DUMMY']*len(days)})
    proc=a.workdir/'BLE_test_processed.csv'; lab=a.workdir/'dummy_test_labels.csv'; testfp=a.workdir/'test_fingerprints.csv'; processed.to_csv(proc,index=False); labels.to_csv(lab,index=False)
    cmd=[sys.executable,str(root/'build_fingerprints.py'),'-i',str(proc),'-l',str(lab),'-o',str(testfp),'--window','10','--step','2','--features','mean,present,std,max,count','--past-context','8','--past-feats','count,std']
    subprocess.run(cmd,check=True)
    tf=pd.read_csv(testfp); Xtest=tf[feat_cols].apply(pd.to_numeric,errors='coerce')
    assert Xtest.shape[1]==175

    args=train_resample.parse_args(['-i','dummy.csv']); args.base='rf'; args.hier='topo2'; args.method='borderline1'; args.target=200; args.seed=42; args.n_jobs=-1
    model=train_resample.build_model(args); model.fit(Xtrain,y); pred=np.asarray(model.predict(Xtest),dtype=str)
    idx=np.asarray([c2i[p] for p in pred],dtype=int)
    groups=model_core.make_groups(tf,'win_start'); timestamps=pd.to_datetime(tf['date'].astype(str)+' '+tf['win_start'].astype(str),errors='coerce').astype('int64').to_numpy()
    session_rows={}
    for g in np.unique(groups):
        rows=np.where(groups==g)[0]; session_rows[int(g)]=rows[np.argsort(timestamps[rows])]
    sm=common.apply_run_fixed(idx,session_rows,c2i,classes); window_pred=np.asarray([classes[int(i)] for i in sm],dtype=object)

    mid=hms_to_seconds(tf['win_start'])+5.0; wdate=tf['date'].astype(str).to_numpy(); out=raw.copy(); rowdate=ts.dt.strftime('%Y-%m-%d').to_numpy(); rowsec=(ts.dt.hour*3600+ts.dt.minute*60+ts.dt.second+ts.dt.microsecond/1e6).to_numpy(float); loc=np.empty(len(out),dtype=object)
    for day in pd.unique(rowdate):
        ri=np.where(rowdate==day)[0]; wi=np.where(wdate==day)[0]; order=np.argsort(mid[wi]); wi=wi[order]; mids=mid[wi]; labs=window_pred[wi]; vals=rowsec[ri]
        right=np.clip(np.searchsorted(mids,vals),0,len(mids)-1); left=np.clip(right-1,0,len(mids)-1); choose=np.abs(vals-mids[left])<=np.abs(vals-mids[right]); loc[ri]=labs[np.where(choose,left,right)]
    out['Location']=loc
    assert len(out)==62222 and out['Location'].notna().all() and set(out['Location']).issubset(set(classes))
    out.to_csv(a.out,index=False); digest=sha256(a.out)
    print('windows',len(tf)); print('rows',len(out)); print('sha256',digest); print('matches_locked_submission',digest==EXPECTED_SHA256); print(out['Location'].value_counts().to_string())
if __name__=='__main__': main()
