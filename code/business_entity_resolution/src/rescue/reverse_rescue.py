# Deep reverse retrieval for India (step 1 of the rescue). From code/business_entity_resolution:
#   python src/rescue/reverse_rescue.py --project ../.. --out ../../work/rescue --country India --mode test
"""Incremental reverse retrieval experiment, leaving baseline artifacts untouched."""
from __future__ import annotations
import argparse,gc,json,os,sys,time
from pathlib import Path
os.environ.setdefault('POLARS_MAX_THREADS','20')
os.environ.setdefault('NUMBA_NUM_THREADS','20')
os.environ.setdefault('OMP_NUM_THREADS','20')
import numpy as np
import polars as pl
from rapidfuzz import fuzz,process

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--country',default='India')
    ap.add_argument('--mode',choices=['val','test'],default='val')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.project/'code/business_entity_resolution'))
    from src.blocking import features_chunked,build_index,query_matrix,_search,_to_long
    from src.features import set_noise_tokens,set_vocab,BLOCK_COLS,SIDE_COLS
    from src.pipeline import add_name_freq,featurize,KEEP_FOR_STAGE2
    from src.stage2 import stage2_frame,STAGE2_COLS
    from src.model import LeaderboardRule,macro_f05
    import lightgbm as lgb
    t0=time.time()
    def log(s):print(f'[{time.time()-t0:.1f}s] {s}',flush=True)
    w=args.project/'work';split='train' if args.mode=='val' else 'test'
    out=args.out/f'{args.mode}_{args.country}';out.mkdir(exist_ok=True)
    arts=w/'artifacts_v9' if args.mode=='val' else args.project/'code/business_entity_resolution/artifacts'
    s1=pl.read_parquet(w/f'{split}_s1_norm.parquet')
    if args.mode=='val':s1=s1.filter(s1['entity_id'].hash(seed=42)%10000>=1900)
    s1=s1.with_row_index('_g').with_columns(pl.col('_g').cast(pl.Int32))
    if args.mode=='val':
        ids=s1['entity_id'].sample(180000,seed=42,shuffle=True)[150000:]
        qrows=s1.filter(pl.col('entity_id').is_in(ids.implode()) & (pl.col('country')==args.country))['_g']
        baseline=pl.read_parquet(w/'val_scored_v9.parquet')
    else:
        ids=s1['entity_id'];qrows=s1.filter(pl.col('country')==args.country)['_g']
        baseline=None
    log(f'loading normalized targets for {split}; {len(qrows)} query owners')
    tg=pl.concat([pl.read_parquet(w/f'{split}_s{i}_norm.parquet') for i in (2,3)])
    tg=tg.with_row_index('_g').with_columns(pl.col('_g').cast(pl.Int32))
    s1,tg=add_name_freq(s1,tg)
    set_noise_tokens(json.loads((arts/'noise_tokens.json').read_text()));set_vocab(s1['name_core'])
    a=s1.filter(pl.col('country')==args.country).select('_g','name_core','addr','business_name').with_row_index('rid')
    b=tg.filter(pl.col('country')==args.country).select('_g','name_core','addr','business_name').with_row_index('rid')
    ag=a['_g'].to_numpy();allowed=np.isin(ag,qrows.to_numpy())
    # Test only expands targets whose current best stage-2 score is below .90.
    # No existing links are removed at retrieval time; competition is resolved later.
    if args.mode=='test':
        confident=(pl.scan_parquet(w/'test_final_scores.parquet').filter(pl.col('p')>=.9).select('t_idx').unique().collect())
        b=b.join(confident.rename({'t_idx':'_g'}),on='_g',how='anti')
        del confident
    candidate_file=out/'new_candidates.parquet'
    if candidate_file.exists():new=pl.read_parquet(candidate_file);log(f'reusing {new.height} new pairs')
    else:
        idx={}
        for ch in ('name','addr'):
            ff=features_chunked(a,ch,chunk=100000)
            idx[ch]=build_index(ff,a.height,10000);del ff;gc.collect()
            log(f'{ch} owner index ready')
        chunks=[]
        for start in range(0,b.height,100000):
            bb=b.slice(start,100000).drop('rid').with_row_index('rid')
            qs=[]
            for ch in ('name','addr'):
                ff=features_chunked(bb,ch)
                qs.append(query_matrix(ff,bb.height,idx[ch],12));del ff
            ii,ss=_search(qs,[idx['name'],idx['addr']],a.height,100)
            # Deeper use of the original scorer preserves original reverse-feature semantics.
            ii[~allowed[np.maximum(ii,0)]]=-1
            cc=_to_long(ii,ss,bb['_g'].to_numpy(),'rev',True)
            cc=cc.with_columns(pl.Series('s1_idx',ag[cc['s1_idx'].to_numpy()]))
            top=pl.DataFrame({'t_idx':bb['_g'],'rev_top1':ss[:,0],'rev_top2':ss[:,1]})
            cc=cc.join(top,on='t_idx')
            si,ti=cc['s1_idx'].to_numpy(),cc['t_idx'].to_numpy()
            aa=s1['addr'].gather(si);ba=tg['addr'].gather(ti)
            noaddr=(ba=='').to_numpy()
            # Cheap observable prefilter; identical retrieval rule for validation and test.
            adr=process.cpdist(aa.to_list(),ba.to_list(),scorer=fuzz.token_set_ratio,workers=20)
            nr=process.cpdist(s1['business_name'].gather(si).to_list(),tg['business_name'].gather(ti).to_list(),scorer=fuzz.ratio,workers=20)
            ok=(~noaddr & (adr>=60)) | (noaddr & (nr>=90))
            cc=cc.filter(pl.Series(ok))
            chunks.append(cc)
            log(f'retrieved {min(start+100000,b.height)}/{b.height} targets; retained {cc.height} pairs this chunk')
            del ii,ss,cc,qs;gc.collect()
        new=pl.concat(chunks);del chunks,idx;gc.collect()
        if args.mode=='val':
            old=(baseline.select('s1_id','t_id').join(s1.select(pl.col('entity_id').alias('s1_id'),pl.col('_g').alias('s1_idx')),on='s1_id')
                 .join(tg.select(pl.col('entity_id').alias('t_id'),pl.col('_g').alias('t_idx')),on='t_id').select('s1_idx','t_idx'))
        else:
            # Restrict cached pairs to owners reached by rescue, streaming only key columns.
            old=pl.scan_parquet(w/'test_cand.parquet').select('s1_idx','t_idx').filter(pl.col('s1_idx').is_in(new['s1_idx'].unique().implode())).collect()
        new=new.unique(['s1_idx','t_idx']).join(old,on=['s1_idx','t_idx'],how='anti')
        new.write_parquet(candidate_file);del old;gc.collect();log(f'{new.height} genuinely new pairs')
    if args.mode=='val':
        gt=(pl.read_csv(args.project/'dataset/train/train_ground_truth.tsv',separator='\t',quote_char=None,infer_schema=False)
            .select(pl.col('source1_entity_id').alias('s1_id'),pl.col('matched_entity_ids').str.split(',').alias('t_id'))
            .explode('t_id').filter(pl.col('t_id').is_not_null() & (pl.col('t_id')!='')))
        vt=gt.filter(pl.col('s1_id').is_in(ids.implode()))
        newids=new.select(pl.Series('s1_id',s1['entity_id'].gather(new['s1_idx'])),pl.Series('t_id',tg['entity_id'].gather(new['t_idx'])))
        recovered=newids.join(vt,on=['s1_id','t_id'])
        log(f'new candidates contain {recovered.height} true missed links')
        recovered.write_parquet(out/'recovered_truth_diagnostic.parquet')
    if new.height==0:return
    counts=pl.scan_parquet(w/('train_cand_150000_30000_d19.parquet' if args.mode=='val' else 'test_cand.parquet')).group_by('s1_idx').len('nold').collect()
    nn=new.group_by('s1_idx').len('nnew')
    new=new.join(counts,on='s1_idx',how='left').join(nn,on='s1_idx').with_columns((pl.col('nold').fill_null(0)+pl.col('nnew')).cast(pl.Int32).alias('s1_ncand')).drop('nold','nnew')
    for col in BLOCK_COLS:
        if col not in new.columns:new=new.with_columns(pl.lit(None,pl.Float32).alias(col))
    booster=lgb.Booster(model_file=str(arts/'lgb_matcher.txt'))
    scored=[]
    for start in range(0,new.height,200000):
        f=featurize(new.slice(start,200000),s1,tg)
        p=booster.predict(f.select(booster.feature_name()).to_numpy().astype(np.float32),num_threads=20)
        scored.append(f.select('s1_idx','t_idx',*KEEP_FOR_STAGE2).with_columns(pl.Series('p',p.astype(np.float32))))
        log(f'scored new stage1 {min(start+200000,new.height)}/{new.height}')
    nf=pl.concat(scored);nf.write_parquet(out/'new_stage1.parquet');del scored,new;gc.collect()
    if args.mode=='test':
        log('test stage-1 scores saved; run src/rescue/partial_rescue.py next');return
    touched=nf['s1_idx'].unique()
    if args.mode=='val':
        oldf=pl.scan_parquet(w/'train_feat_150000_30000_d19.parquet').filter(pl.col('s1_idx').is_in(touched.implode())).collect()
        p=booster.predict(oldf.select(booster.feature_name()).to_numpy().astype(np.float32),num_threads=20)
        oldf=oldf.select('s1_idx','t_idx',*KEEP_FOR_STAGE2).with_columns(pl.Series('p',p.astype(np.float32)))
    else:
        oldf=pl.scan_parquet(w/'test_scores_s1/*.parquet').filter(pl.col('s1_idx').is_in(touched.implode())).collect()
    full=pl.concat([oldf,nf],how='vertical_relaxed');del nf,oldf;gc.collect()
    b2=lgb.Booster(model_file=str(arts/'lgb_stage2.txt'))
    tn,ta=tg['name_core'].to_numpy(),tg['addr'].to_numpy();is3=tg['entity_id'].str.starts_with('S3-').to_numpy()
    final=[]
    for start in range(0,len(touched),5000):
        d=full.filter(pl.col('s1_idx').is_in(touched.slice(start,5000).implode()))
        f=stage2_frame(d,tn,ta,is3)
        p=b2.predict(f.select(STAGE2_COLS).to_numpy(),num_threads=20)
        final.append(f.select(pl.col('g').alias('s1_idx'),pl.col('t').alias('t_idx')).with_columns(pl.Series('p',p.astype(np.float32))))
        log(f'stage2 {min(start+5000,len(touched))}/{len(touched)} owners')
    final=pl.concat(final);final.write_parquet(out/'affected_scores.parquet')
    if args.mode=='test':
        log('test affected scores saved; application and validation are separate steps');return
    base=(baseline.join(s1.select(pl.col('entity_id').alias('s1_id'),pl.col('_g').alias('s1_idx')),on='s1_id')
          .join(tg.select(pl.col('entity_id').alias('t_id'),pl.col('_g').alias('t_idx')),on='t_id').select('s1_idx','t_idx','p'))
    revised=pl.concat([base.filter(~pl.col('s1_idx').is_in(touched.implode())),final],how='vertical_relaxed')
    na=tg.select(pl.col('_g').alias('t_idx'),(pl.col('addr')=='').alias('noaddr'))
    rule=LeaderboardRule()
    def pred(sc):
        z=rule.apply(sc,na)
        # Deterministic exclusivity for exact ties, and per-source quotas.
        z=z.join(sc,on=['s1_idx','t_idx']).sort(['p','s1_idx'],descending=[True,False]).unique('t_idx',keep='first',maintain_order=True)
        z=z.with_columns(pl.Series('src3',tg['entity_id'].gather(z['t_idx']).str.starts_with('S3-')))
        z=z.with_columns(pl.col('p').rank('ordinal',descending=True).over('s1_idx','src3').alias('r'))
        z=z.filter(pl.col('r')<=pl.when(pl.col('src3')).then(6).otherwise(5))
        return z.select(pl.Series('s1_id',s1['entity_id'].gather(z['s1_idx'])),pl.Series('t_id',tg['entity_id'].gather(z['t_idx'])))
    bp,rp=pred(base),pred(revised)
    bp.write_parquet(out/'baseline_selected.parquet');rp.write_parquet(out/'revised_selected.parquet')
    summary={'country':args.country,'mode':args.mode,'seconds':time.time()-t0,'recovered_candidates':recovered.height,'metrics':{}}
    for label,vids in [('all',ids),('half_A',ids[:15000]),('half_B',ids[15000:])]:
        bs=macro_f05(bp,vt,vids);rs=macro_f05(rp,vt,vids)
        summary['metrics'][label]={'baseline':bs,'revised':rs,'delta':rs-bs}
    added=rp.join(bp,on=['s1_id','t_id'],how='anti');removed=bp.join(rp,on=['s1_id','t_id'],how='anti')
    summary.update(added=added.height,added_true=added.join(vt,on=['s1_id','t_id']).height,removed=removed.height,removed_true=removed.join(vt,on=['s1_id','t_id']).height)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));log(json.dumps(summary))

if __name__=='__main__':main()
