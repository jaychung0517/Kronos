"""Research only: no messaging or orders. Historical reruns are retrospective."""
import os
import hashlib
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import yfinance as yf
from model import Kronos, KronosTokenizer, KronosPredictor
from model.kronos import calc_time_stamps
from scripts import taiwan_forecast as tw
from scripts.us_close_forecast import future_us_dates
OUT=Path("bias_audit_outputs")
OUT.mkdir(exist_ok=True)
TW_RUNS=[42593955187,42791282917,42995748322,43192072669,43401886242,43840368536,44056212940,44276566218,44497053576,44711303202,45147659372]
LOGGED={"NASDAQ":[1.38,1.82,2.01,1.68,1.73,2.81,2.65,2.90,4.66,7.76,3.82],"SOX":[4.68,4.57,4.77,5.39,5.21,3.33,7.54,8.35,8.45,16.78,4.50]}
def archive():
    rows=[]
    for run in TW_RUNS:
        dest=OUT/"archives"/str(run)
        subprocess.run(["gh","run","download",str(run),"-n","kronos-taiwan-v3-results","-D",str(dest)],check=True)
        files=list(dest.rglob("v33_reversal_live.csv"))
        if len(files)!=1: raise ValueError(f"missing or ambiguous Taiwan file {run}")
        f=pd.read_csv(files[0])
        if not {"market","origin","forecast_3d_pct"}.issubset(f): raise ValueError("unexpected schema")
        f["run_id"]=run
        rows.append(f)
    pd.concat(rows,ignore_index=True).to_csv(OUT/"taiwan_archived.csv",index=False)
def controlled_predict(predictor, h, norm, dates):
    def matrix(frame):
        q=frame.copy()
        q["volume"]=0.0
        q["amount"]=0.0
        return q[predictor.price_cols+["volume","amount"]].to_numpy(dtype=np.float32)
    x=matrix(h)
    z=matrix(norm)
    mu=z.mean(axis=0)
    sd=z.std(axis=0)+1e-5
    scaled=np.clip((x-mu)/sd,-predictor.clip,predictor.clip)
    pred=predictor.generate(scaled[None],calc_time_stamps(pd.Series(h.index)).to_numpy(dtype=np.float32)[None],calc_time_stamps(pd.Series(dates)).to_numpy(dtype=np.float32)[None],5,1.,0,.9,5,False)
    return pd.DataFrame(pred[0]*sd+mu,columns=predictor.price_cols+["volume","amount"])

def main():
    # Round one archives remain immutable in run 45317540936.
    model=Kronos.from_pretrained("NeoQuasar/Kronos-base")
    tokenizer=KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model.eval();tokenizer.eval()
    predictor=KronosPredictor(model,tokenizer,device="cpu",max_context=512)
    rows=[]; diagnostic=[]
    # Round 4: independent Taiwan index replication, fixed retrospective cutoff.
    for market,symbol in {"TAIEX":"^TWII"}.items():
        f=yf.download(symbol,start="2024-01-01",end="2026-11-24",auto_adjust=False,progress=False,multi_level_index=False)
        f.columns=[str(c).lower() for c in f.columns]
        f=f.sort_index()
        if f.index.tz is not None: f.index=f.index.tz_localize(None)
        f.index=f.index.normalize()
        if f.index.has_duplicates: raise ValueError("duplicate dates")
        f.to_csv(OUT/f"{market}_input.csv")
        print("input SHA256",market,hashlib.sha256((OUT/f"{market}_input.csv").read_bytes()).hexdigest())
        tw.validate_ohlc(f,market)
        origins=f.loc["2026-11-09":"2026-11-23"].index
        if len(origins)<8: raise ValueError("insufficient origins")
        for k,origin in enumerate(origins):
            hist=f.loc[:origin]
            dates,calendar=tw.future_market_dates(origin)
            if calendar!="csv_calendar": raise ValueError("official calendar required")
            ref=float(hist.close.iloc[-1])
            gap=(hist.close.tail(200).mean()/ref-1)*100
            diagnostic.append(dict(market=market,origin=origin,mean_gap_pct=gap))
            for group,n,volume in [("L200",200,False),("L400",400,False),("L200N400",200,False),("L400N200",400,False)]:
                h=hist.tail(n).copy()
                if len(h)!=n: raise ValueError("insufficient history")
                if volume and ("volume" not in h or not np.isfinite(h.volume).all() or not (h.volume>0).all()):
                    rows.append(dict(market=market,origin=origin,group=group,status="volume unavailable"))
                    continue
                for seed_i in range(3):
                    seed=tw.experiment_seed(market,origin,seed_i)+30000
                    tw.seed_everything(seed)
                    cols=tw.OHLC+(["volume"] if volume else [])
                    raw=controlled_predict(predictor,h,hist.tail(int(group.split("N")[1]) if "N" in group else n),dates)
                    if "N" not in group:
                        tw.seed_everything(seed)
                        control=predictor.predict(h[cols].reset_index(drop=True),pd.Series(h.index),pd.Series(dates),5,T=1.,top_p=.9,sample_count=5,verbose=False)
                        np.testing.assert_allclose(raw[tw.OHLC].to_numpy(),control[tw.OHLC].to_numpy(),rtol=1e-5,atol=1e-4)
                    if len(raw)!=5 or not np.isfinite(raw.close).all() or not (raw.close>0).all(): raise ValueError("invalid predicted close")
                    valid=(raw.high>=raw[["open","close","low"]].max(axis=1))&(raw.low<=raw[["open","close","high"]].min(axis=1))
                    for step in [1,3,5]:
                        date=dates[step-1]
                        actual=(float(f.loc[date,"close"])/ref-1)*100 if date in f.index else np.nan
                        rows.append(dict(market=market,origin=origin,group=group,seed=seed_i,horizon=step,target=date,pred_pct=(float(raw.close.iloc[step-1])/ref-1)*100,actual_pct=actual,momentum_pct=(ref/float(hist.close.iloc[-step-1])-1)*100,mean_reversion_pct=gap,ohlc_valid=bool(valid.iloc[step-1]),status="ok"))
    r=pd.DataFrame(rows); r.to_csv(OUT/"seed_results.csv",index=False)
    d=pd.DataFrame(diagnostic);d.to_csv(OUT/"mean_gap.csv",index=False)
    # No original US forecast values are reused for Taiwan.
    a=r[r.status=="ok"].groupby(["market","origin","group","horizon"],as_index=False).agg(pred_pct=("pred_pct","mean"),median_pct=("pred_pct","median"),actual_pct=("actual_pct","first"),valid_rate=("ohlc_valid","mean"),momentum_pct=("momentum_pct","first"),mean_reversion_pct=("mean_reversion_pct","first"))
    a.to_csv(OUT/"averaged.csv",index=False)
    a=a.dropna(subset=["actual_pct"])
    a["hit"]=np.sign(a.pred_pct)==np.sign(a.actual_pct)
    a["up"]=a.pred_pct>0
    a["always_up"]=a.actual_pct>0
    a["always_down"]=a.actual_pct<0
    a["error"]=a.pred_pct-a.actual_pct
    a["abs_error"]=a.error.abs()
    a["zero_mae"]=a.actual_pct.abs()
    a["extreme"]=a.pred_pct.abs()>10
    a["median_hit"]=np.sign(a.median_pct)==np.sign(a.actual_pct)
    a["momentum_hit"]=np.sign(a.momentum_pct)==np.sign(a.actual_pct)
    a["mean_reversion_hit"]=np.sign(a.mean_reversion_pct)==np.sign(a.actual_pct)
    a["up_recall"]=a.hit.where(a.actual_pct>0)
    a["down_recall"]=a.hit.where(a.actual_pct<0)
    a.groupby(["market","group","horizon"]).agg(n=("hit","size"),momentum_accuracy=("momentum_hit","mean"),mean_reversion_accuracy=("mean_reversion_hit","mean"),accuracy=("hit","mean"),up_fraction=("up","mean"),always_up=("always_up","mean"),always_down=("always_down","mean"),mean_error=("error","mean"),mae=("abs_error","mean"),zero_mae=("zero_mae","mean"),extreme_rate=("extreme","mean"),median_accuracy=("median_hit","mean"),up_recall=("up_recall","mean"),down_recall=("down_recall","mean"),valid_rate=("valid_rate","mean")).to_csv(OUT/"metrics.csv")
    print((OUT/"metrics.csv").read_text())
    print("retrospective Taiwan index only; TX requires as-of contract reconstruction")
if __name__=="__main__":
    main()
