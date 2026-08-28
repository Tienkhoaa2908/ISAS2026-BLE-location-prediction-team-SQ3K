#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
DLCS=REPO/'src'/'dlcs'
sys.path.insert(0,str(DLCS))
import evaluation_core as CORE
COMMON=CORE.COMMON

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--outdir',type=Path,required=True); ap.add_argument('--specialist-trees',type=int,default=1200); ap.add_argument('--force',action='store_true')
    a=ap.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    class A: pass
    x=A(); x.data_root=a.data_root.resolve(); x.data_zip=None; x.outdir=a.outdir
    root=CORE.prepare_dataset(x); train_resample=CORE.import_train_resample(root)
    sparse=COMMON.load_sparse_ble(root); data=COMMON.build_bridge_data(root,sparse,__import__('model_core'))
    y=data['y'].astype(str).to_numpy(); dates=data['frame']['date'].astype(str).to_numpy(); classes=sorted(np.unique(y).tolist()); days=sorted(np.unique(dates).tolist())
    parts=[]; rows=[]
    for day in days:
        tr=np.where(dates!=day)[0]; te=np.where(dates==day)[0]
        cache=a.outdir/'prediction_cache'/f'lodo_{day.replace("-","")}_sp{a.specialist_trees}.npz'
        r=CORE.fit_predict_models(data,train_resample,tr,te,classes,a.specialist_trees,cache,a.force)
        base=r['base_pred'].astype(str)
        hard,audit=CORE.apply_v29_hard_gate(data,te,base,r['base_proba'],r['spec_proba'],classes,0.75)
        systems={
          'main_raw':base,
          'dlcs_raw':hard,
          'main_centered':CORE.exact_centered_smoothing(base,data,te,classes),
          'dlcs_centered':CORE.exact_centered_smoothing(hard,data,te,classes),
        }
        for name,pred in systems.items():
            m=CORE.fixed_metrics(y[te],pred,classes); rows.append({'held_out_day':day,'system':name,'rows':len(te),'macro_f1':m['macro_f1'],'accuracy':m['accuracy'],'balanced_accuracy':m['balanced_accuracy'],'zero_f1':m['zero_f1']})
        f=data['frame'].iloc[te][['date','win_start','win_end','room']].copy().reset_index(drop=True).rename(columns={'room':'true'})
        for name,pred in systems.items(): f[name]=pred
        parts.append(f)
        print(day,{k:round(CORE.fixed_metrics(y[te],v,classes)['macro_f1'],6) for k,v in systems.items()})
    pred_df=pd.concat(parts,ignore_index=True); pred_df.to_csv(a.outdir/'paper_lodo_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(a.outdir/'paper_lodo_by_day.csv',index=False)
    agg=[]; yagg=pred_df['true'].astype(str).to_numpy()
    for name in ['main_raw','dlcs_raw','main_centered','dlcs_centered']:
        p=pred_df[name].astype(str).to_numpy(); m=CORE.fixed_metrics(yagg,p,classes); agg.append({'system':name,'macro_f1':m['macro_f1'],'accuracy':m['accuracy'],'balanced_accuracy':m['balanced_accuracy'],'zero_f1':m['zero_f1'],'rows':len(yagg)})
        per=CORE.f1_score(yagg,p,labels=classes,average=None,zero_division=0)
        pd.DataFrame({'room':classes,'f1':per}).to_csv(a.outdir/f'per_class_{name}.csv',index=False)
        cm=confusion_matrix(yagg,p,labels=classes,normalize='true'); pd.DataFrame(cm,index=classes,columns=classes).to_csv(a.outdir/f'confusion_{name}_normalized.csv')
    pd.DataFrame(agg).to_csv(a.outdir/'paper_lodo_aggregate.csv',index=False)
    print(pd.DataFrame(agg).to_string(index=False))
if __name__=='__main__': main()
