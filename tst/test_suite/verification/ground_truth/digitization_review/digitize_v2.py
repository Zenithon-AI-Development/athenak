#!/usr/bin/env python3
"""v2: cull spurious points flagged by vision QC; tighten shock + B4 masks.
Outputs cleaned extracted_points_v2.json + clean verification plots (extracted
points on labeled axes) + refreshed overlays for B3-shock and B4."""
import json, numpy as np, os
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
FIGDIR="figures"; OUT="extracted"; os.makedirs(OUT,exist_ok=True)
def load(fn): return np.asarray(Image.open(os.path.join(FIGDIR,fn)).convert("RGB")).astype(int)
def lin(px,p0,v0,p1,v1): return v0+(px-p0)*(v1-v0)/(p1-p0)
def clusters(mask,min_size=6):
    lab,n=ndimage.label(mask); out=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        if len(xs)>=min_size: out.append((xs.mean(),ys.mean(),len(xs),xs.min(),xs.max(),ys.min(),ys.max()))
    return out
R={}

# A) B3 growth: magenta circles. Cull the spurious low-x/high-y blob (legend/overlap).
fn="knapp_rm_growth_rate.png"; img=load(fn)
X=lambda px:lin(px,357.5,0.0,1971.7,10.0); Y=lambda py:lin(py,295.9,4.5,1461.0,1.0)
cl=clusters((np.abs(img-np.array([255,0,255])).sum(2)<120),20)
pts=sorted([(X(cx),Y(cy)) for cx,cy,*_ in cl])
# physical: growth rises with displacement; drop points that violate (x<2 and y>3)
pts=[(x,y) for x,y in pts if not (x<2 and y>3) and 0.7<y<5.2 and -0.5<x<12]
R["B3_rm_growth_factor"]={"figure":fn,"x_axis":"k_z*delta_r (dimensionless)","y_axis":"growth factor h/h0 (dimensionless)","series":"experiment (magenta 'Z Data')","qc":"GOOD","points":[{"x":round(x,3),"y":round(y,3)} for x,y in pts]}

# D) B1 amplitude: blue '+', log mm. Cull impossible early high-amp blob.
fn="sinars_mrt_growth_rate.png"; img=load(fn)
X=lambda px:lin(px,99.5,0.0,530.0,60.0)
def Yl(py):return 10**lin(py,28.5,0.0,358.5,-2.0)
r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
cl=clusters((b>150)&(r<90)&(g<90),8)
pts=sorted([(X(cx),Yl(cy)) for cx,cy,*_ in cl])
pts=[(x,y) for x,y in pts if -2<x<90 and 1e-3<y<3 and not (x<10 and y>0.1)]
R["B1_single_mode_amplitude"]={"figure":fn,"x_axis":"time (ns)","y_axis":"mode amplitude (mm, log)","series":"experiment (blue '+')","qc":"GOOD","points":[{"t_ns":round(x,2),"amp_mm":round(y,4)} for x,y in pts]}

# E) B5 BR: red circles. (clean already)
fn="z2977_rb.png"; img=load(fn)
X=lambda px:lin(px,74.5,0.25,512.4,1.75); Y=lambda py:lin(py,22.4,0.6,323.0,0.1)
r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
cl=clusters((r>150)&(g<90)&(b<90),12)
pts=sorted([(X(cx),Y(cy)) for cx,cy,*_ in cl])
pts=[(x,y) for x,y in pts if 0.0<x<2.2 and 0.05<y<0.7]
R["B5_BR_vs_preheat"]={"figure":fn,"x_axis":"laser preheat energy (kJ)","y_axis":"BR (MG*cm)","series":"experiment (red 'Data')","qc":"GOOD","points":[{"preheat_kJ":round(x,3),"BR_MGcm":round(y,3)} for x,y in pts]}

# B) B3 trajectory liner (blue solid) -- clean; shock (green) -- per-column median + outlier cull
fn="knapp_rm_r_vs_t.png"; img=load(fn)
Xt=lambda px:lin(px,83.4,2850.0,626.2,3150.0); Yr=lambda py:lin(py,44.5,5.0,276.7,1.0)
r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
def trace_median(mask,xmin_px=84,xmax_px=626):
    cols={}; ys,xs=np.where(mask)
    for x,y in zip(xs,ys):
        if xmin_px<=x<=xmax_px: cols.setdefault(x,[]).append(y)
    out=[]
    for x in sorted(cols):
        v=cols[x]
        if len(v)>=2:  # require >=2 px to reject single-px green antialias
            out.append((Xt(x),Yr(np.median(v))))
    return out
liner=trace_median((b>120)&(r<90)&(g<120))
# shock: stricter green, and physical radius window 0<r<5; then sort + downsample
shock_raw=trace_median((g>120)&(r<100)&(b<100))
shock=[(t,rr) for t,rr in shock_raw if 0.0<rr<5.1]
def ds(p,n=30):
    if len(p)<=n: return p
    idx=np.linspace(0,len(p)-1,n).astype(int); return [p[i] for i in idx]
liner=ds(liner); shock=ds(shock)
R["B3_trajectory_liner_exp"]={"figure":fn,"x_axis":"time (ns)","y_axis":"inner liner radius (mm)","series":"experiment PDV (blue)","qc":"GOOD","points":[{"t_ns":round(x,1),"r_mm":round(y,3)} for x,y in liner]}
R["B3_trajectory_shock_exp"]={"figure":fn,"x_axis":"time (ns)","y_axis":"shock radius (mm)","series":"experiment PDV (green)","qc":"PARTIAL - green line broken/dashed; physical cull applied","points":[{"t_ns":round(x,1),"r_mm":round(y,3)} for x,y in shock]}

# C) B4 panel(b): isolate gray OPEN circles. Strategy: gray ring pixels = grayish AND
# local (exclude orange/blue). Then keep clusters that look like ring (bbox roughly square, modest size).
fn="knapp_confine_r_vs_t.png"; img=load(fn)
Xb=lambda px:lin(px,833.1,3115.0,1419.0,3140.0); Yb=lambda py:lin(py,37.6,0.65,422.4,0.30)
r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
gray=(np.abs(r-g)<22)&(np.abs(g-b)<22)&(np.abs(r-b)<22)&(r>60)&(r<175)
panel=np.zeros_like(gray); panel[35:425,835:1418]=True
cl=clusters(gray&panel,12)
b4=[]
for cx,cy,sz,x0,x1,y0,y1 in cl:
    w=x1-x0+1; h=y1-y0+1
    # ring marker: roughly square-ish, not a long vertical errbar-only sliver, not a text glyph row
    if 6<=w<=34 and 6<=h<=60 and 0.4<w/h<2.5 and sz<700:
        b4.append((Xb(cx),Yb(cy),Yb(y1),Yb(y0)))  # y1=bottom(lower r), y0=top(higher r)
b4=sorted(b4)
b4=[d for d in b4 if 3113<d[0]<3142 and 0.30<d[1]<0.66]
R["B4_inner_radius_points_exp"]={"figure":fn+" (panel b)","x_axis":"time (ns)","y_axis":"inner liner radius (mm)","series":"experiment radiograph (gray open circles + errbars)","qc":"LOW - dense overlap w/ FLASH markers; treat as approximate","points":[{"t_ns":round(t,1),"r_mm":round(rr,3),"err_lo_mm":round(elo,3),"err_hi_mm":round(ehi,3)} for t,rr,elo,ehi in b4]}

json.dump(R,open(os.path.join(OUT,"extracted_points_v2.json"),"w"),indent=2)
print("v2 counts:")
for k,v in R.items(): print(f"  {k}: {len(v['points'])} pts  [{v['qc']}]")

# ---- clean verification plots (extracted pts on labeled axes) ----
def vplot(key,xk,yk,logy=False,inv=False):
    d=R[key]; P=d["points"]
    xs=[p[xk] for p in P]; ys=[p[yk] for p in P]
    fig,ax=plt.subplots(figsize=(6,4.2))
    ax.plot(xs,ys,'o-' if 'traj' in key else 'o',color='crimson',ms=6)
    if logy: ax.set_yscale('log')
    if inv: ax.invert_yaxis()
    ax.set_xlabel(d["x_axis"]); ax.set_ylabel(d["y_axis"]); ax.grid(alpha=.3)
    ax.set_title(f"{key}  [{d['qc']}]  n={len(P)}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"clean_{key}.png"),dpi=110); plt.close(fig)
vplot("B3_rm_growth_factor","x","y")
vplot("B1_single_mode_amplitude","t_ns","amp_mm",logy=True)
vplot("B5_BR_vs_preheat","preheat_kJ","BR_MGcm")

# refreshed overlays for the two reworked series
def overlay_traj():
    fn="knapp_rm_r_vs_t.png"; fig,ax=plt.subplots(figsize=(7,5)); ax.imshow(Image.open(os.path.join(FIGDIR,fn)))
    for key,c in [("B3_trajectory_liner_exp",'cyan'),("B3_trajectory_shock_exp",'lime')]:
        P=R[key]["points"]
        px=[83.4+(p["t_ns"]-2850)*(626.2-83.4)/300 for p in P]
        py=[44.5+(p["r_mm"]-5)*(276.7-44.5)/(1-5) for p in P]
        ax.plot(px,py,'.',color=c,ms=5,label=key.split('_')[2])
    ax.legend(); ax.set_title("B3 trajectory v2 — liner(cyan)/shock(lime)")
    fig.savefig(os.path.join(OUT,"B3_trajectory_overlay_v2.png"),dpi=110,bbox_inches='tight'); plt.close(fig)
def overlay_b4():
    fn="knapp_confine_r_vs_t.png"; fig,ax=plt.subplots(figsize=(8,4)); ax.imshow(Image.open(os.path.join(FIGDIR,fn)))
    P=R["B4_inner_radius_points_exp"]["points"]
    px=[833.1+(p["t_ns"]-3115)*(1419-833.1)/25 for p in P]
    py=[37.6+(p["r_mm"]-0.65)*(422.4-37.6)/(0.30-0.65) for p in P]
    ax.scatter(px,py,s=130,facecolors='none',edgecolors='red',linewidths=1.8)
    ax.set_title("B4 inner-radius v2 [LOW confidence]")
    fig.savefig(os.path.join(OUT,"B4_inner_radius_overlay_v2.png"),dpi=110,bbox_inches='tight'); plt.close(fig)
overlay_traj(); overlay_b4()
print("wrote clean plots + v2 overlays")
