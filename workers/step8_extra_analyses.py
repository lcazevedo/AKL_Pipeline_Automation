"""workers/step8_extra_analyses.py — run: ./run_worker.sh workers/step8_extra_analyses.py
Extra Step-8 analyses for questions 1-11 (topic lifecycle, pioneers, app x method, evidence vs volume, directions, trajectory)."""
import glob, numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
import os
ROOT=str(Path(os.environ.get("AKL_WORKSPACE") or Path.cwd()))+"/"
c=pd.read_pickle(ROOT+"data/02_intermediate/step5_corpus_filtered_relevant.pkl")
c["k"]=c.doi.astype(str).str.lower().str.strip()
f=glob.glob(ROOT+"data/05_exports/step6_topics_final_layer_*_selected.xlsx")[0]
tm=[]
for n,d in pd.read_excel(f,sheet_name=None).items():
    if n in("Topics","All_Layers") or "doi" not in d.columns: continue
    tm.append(pd.DataFrame({"k":d.doi.astype(str).str.lower().str.strip(),"topic":n}))
tm=pd.concat(tm).drop_duplicates("k")
c=c.merge(tm,on="k",how="left"); c=c[c.topic.notna()&c.year.between(2000,2025)].copy()
e=pd.read_pickle(ROOT+"data/02_intermediate/step7_extractions.pkl"); e["k"]=e.doi.astype(str).str.lower().str.strip()
E={v:e[e.vector==v] for v in e.vector.unique()}
yrs=list(range(2000,2026)); out=[]
w=out.append
tot=c.groupby("year").size().reindex(yrs,fill_value=0)
w(f"# Step 8 — extra analyses (questions 1-11)\n\nCorpus: {len(c)} papers with a Step-6 topic, 2000-2025 (2026 excluded: partial year). Whole-field growth: {tot[2015]} papers in 2015 -> {tot[2025]} in 2025 (x{tot[2025]/tot[2015]:.1f}). Because the whole field grows fast, topic dynamics are judged by **share of the yearly corpus**, not raw counts.\n")

# ---- Q1/Q11 topic lifecycle + trajectory
rows=[]
for t,g in c.groupby("topic"):
    y=g.groupby("year").size().reindex(yrs,fill_value=0); sh=y/tot
    first=next((yy for yy in yrs if y[yy]>=5),None)
    pre2010=int(y.loc[:2009].sum())
    win=[sh.loc[a_:b_].mean() for a_,b_ in ((2014,2016),(2017,2019),(2020,2022),(2023,2025))]
    xs=np.arange(2015,2026); ys=np.log(y.loc[2015:2025].clip(lower=1).values)
    sl=stats.linregress(xs,ys)
    gr=lambda k:100*(np.exp(k)-1)
    # regime from the share in four 3-year windows (2014-16, 2017-19, 2020-22, 2023-25)
    if win[3]<=win[0]*0.8: reg="declining share"                                       # clearly below the 2014-16 level
    elif win[3]>=win[0]*1.25 and win[3]>=win[2]*1.10: reg="rising share"                # above 2014-16 and still growing
    elif max(win[1:])>=win[0]*1.25 and win[3]<0.9*max(win[1],win[2]): reg="peaked, share falling"
    else: reg="stable share"
    rows.append(dict(topic=t,n=len(g),first_year_5p=first,pre2010=pre2010,
        share14_16=round(100*win[0],2),share17_19=round(100*win[1],2),share20_22=round(100*win[2],2),share23_25=round(100*win[3],2),
        growth_yr_pct=round(gr(sl.slope),1),growth_lo=round(gr(sl.slope-1.96*sl.stderr),1),growth_hi=round(gr(sl.slope+1.96*sl.stderr),1),share_trend=reg))
T=pd.DataFrame(rows).sort_values("n",ascending=False)
w("## A. Topic lifecycle and trajectory (Q1, Q11)\nshare* = % of the yearly corpus (3-year means). growth_yr_pct = log-linear fit of yearly counts 2015-2025 with a 95% interval for the slope (growth_lo/hi). Whole-field growth is ~35%/yr, so growth_yr_pct above that means gaining share. This is a trend description, not a forecast.\n")
w(T.to_markdown(index=False)+"\n")
w("**Top 5 gaining share:** "+", ".join(T.assign(d=T.share23_25-T.share14_16).sort_values("d",ascending=False).topic.head(5))+"\n")
w("**Top 5 losing share:** "+", ".join(T.assign(d=T.share23_25-T.share14_16).sort_values("d").topic.head(5))+"\n")

# ---- Q2 pioneers
c["early"]=c.year<=2012
ct=c.assign(country=c.first_country.fillna("Unknown")).groupby(["country","early"]).size().unstack(fill_value=0)
ct["early_share"]=100*ct[True]/ct[True].sum(); ct["recent_share"]=100*ct[False]/ct[False].sum()
ct=ct.assign(total=ct[True]+ct[False]).sort_values("total",ascending=False).head(12)
w("## B. Pioneers (Q2)\nShare of early papers (<=2012) vs later papers by first-author country.\n")
w(ct[["total","early_share","recent_share"]].round(1).to_markdown()+"\n")
pio=c[c.year<=2010].groupby("topic").size().sort_values(ascending=False)
w("Topics with papers before 2011 (count): "+"; ".join(f"{k}={v}" for k,v in pio.head(12).items())+"\n")

# ---- Q3/Q10 app x method
a=E["application_task"][["k","macro_category"]].drop_duplicates("k").rename(columns={"macro_category":"app"})
m=E["ml_methods"][["k","macro_category"]].drop_duplicates("k").rename(columns={"macro_category":"method"})
am=c[["k","year"]].merge(a,on="k").merge(m,on="k")
X=pd.crosstab(am.app,am.method)
exp=np.outer(X.sum(1),X.sum(0))/X.values.sum()
ratio=pd.DataFrame(X.values/exp,index=X.index,columns=X.columns)
w("## C. Application x method (Q3, Q10)\nCounts (papers).\n"); w(X.to_markdown()+"\n")
low=[(X.index[i],X.columns[j],int(X.iloc[i,j]),round(exp[i,j],1)) for i in range(X.shape[0]) for j in range(X.shape[1]) if exp[i,j]>=30 and X.iloc[i,j]<0.5*exp[i,j]]
low=sorted(low,key=lambda r:r[2]/r[3])[:12]
w("**Under-represented pairs (observed < 50% of expected, expected >= 30):** observed vs expected\n"+"\n".join(f"- {a_} x {b_}: {o} vs {x}" for a_,b_,o,x in low)+"\n")
mm=pd.crosstab(am.year,am.method,normalize="index").loc[2005:2025]*100
top=am.method.value_counts().head(6).index
w("Method mix by year, % of papers (top 6 method categories), selected years.\n"); w(mm.loc[[2005,2010,2015,2018,2020,2022,2024,2025],top].round(1).to_markdown()+"\n")

# ---- Q4/Q7 evidence vs volume
def share(vec,pred):
    d=E[vec][["k","macro_category"]].drop_duplicates("k"); d=d[d.k.isin(c.k)].merge(c[["k","topic"]],on="k"); d["hit"]=pred(d.macro_category)
    return d.groupby("topic").hit.mean()*100
ev=pd.DataFrame({"exp_validation_%":share("validation",lambda s:s=="Experimental Validation"),
    "no_or_cv_only_%":share("validation",lambda s:s.isin(["No Validation","Cross-Validation Only"])),
    "dft_sim_data_%":share("data_source",lambda s:s=="Dft Or Simulation"),
    "experimental_data_%":share("data_source",lambda s:s=="Experiments")})
bd=E["benchmarks_datasets"].k.drop_duplicates(); bd=c.assign(b=c.k.isin(bd)).groupby("topic").b.mean()*100
ev["named_benchmark_%"]=bd
ev=ev.join(T.set_index("topic")[["n","growth_yr_pct","share_trend"]]).sort_values("growth_yr_pct",ascending=False)
w("## D. Evidence vs volume (Q4, Q7)\n"); w(ev.round(1).to_markdown()+"\n")
r=stats.spearmanr(ev["growth_yr_pct"],ev["exp_validation_%"],nan_policy="omit")
w(f"Spearman(growth rate, experimental-validation share) across topics: rho={r.correlation:.2f}, p={r.pvalue:.3f}\n")
cit=c.groupby("topic").citations_per_year.median().round(2)
w("Median citations/year by topic: "+"; ".join(f"{k}={v}" for k,v in cit.sort_values(ascending=False).items())+"\n")

# ---- Q9 directions
d=E["directions"][["k","macro_category","micro_label"]].merge(c[["k","year","topic"]],on="k")
dy=pd.crosstab(d.macro_category,d.year)
rec=dy.loc[:,2023:2025].sum(1)/dy.loc[:,2023:2025].sum().sum(); old=dy.loc[:,2015:2020].sum(1)/dy.loc[:,2015:2020].sum().sum()
dd=pd.DataFrame({"total":dy.sum(1),"share_2023_25_%":rec*100,"share_2015_20_%":old*100}); dd["ratio"]=dd["share_2023_25_%"]/dd["share_2015_20_%"]
w("## E. Future directions (Q9)\nShare of direction mentions by macro-category, recent vs earlier.\n"); w(dd.sort_values("total",ascending=False).round(2).to_markdown()+"\n")
kw={"LLM / language models":r"\b(llm|large language|language model|chatgpt|gpt)","autonomous / self-driving lab":r"autonom|self-driving|closed-loop|robot",
 "inverse design / generative":r"inverse design|generative|diffusion","transfer learning / foundation":r"transfer learning|foundation|pre-?train",
 "interpretability / explainable":r"interpret|explainab","uncertainty quantification":r"uncertaint","active learning / Bayesian opt.":r"active learning|bayesian optim",
 "physics-informed / hybrid":r"physics-informed|physics-based|hybrid","multi-scale / multimodal":r"multi-?scale|multi-?modal|multi-?fidelity","data / benchmarks":r"benchmark|standard|database|dataset|data sharing"}
ml=d.micro_label.astype(str).str.lower(); docs=d.groupby("year").k.nunique()
rows=[]
for n,rx in kw.items():
    h=d[ml.str.contains(rx,regex=True)]
    s=h.groupby("year").k.nunique().reindex(range(2010,2026),fill_value=0)
    rows.append([n,len(h)]+[round(100*s[y]/docs[y],1) for y in (2015,2018,2020,2022,2024,2025)])
w("Share (%) of papers whose extracted direction labels match each theme:\n"); w(pd.DataFrame(rows,columns=["theme","mentions","2015","2018","2020","2022","2024","2025"]).to_markdown(index=False)+"\n")
ch=E["application_challenges"][["k","macro_category"]].merge(c[["k","topic"]],on="k")
w("Top domain challenges overall: "+"; ".join(f"{k} ({v})" for k,v in ch.macro_category.value_counts().head(8).items())+"\n")
open(ROOT+"data/04_reports/step8_extra_analyses.md","w").write("\n".join(out))
pd.to_pickle({"T":T,"ev":ev,"X":X,"exp":exp,"am":am,"tot":tot,"dd":dd,"cit":cit}, ROOT+"data/02_intermediate/step8_extra_tables.pkl")   # reused by step9_extra_figures.py
print("ok",len(out))
