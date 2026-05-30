#!/usr/bin/env python3
"""Color-keyed digitization of the experimental series in the Ellison-paper figures
(arXiv:2504.10760 source images), with overlay plots for human verification.

For each figure: load the source PNG, select pixels matching the EXPERIMENT series
color, cluster them into markers, map pixel->data via the (vision-calibrated) axis
anchors, and write (a) an overlay PNG (extracted markers re-drawn on the original)
and (b) a JSON of extracted points. Lines are traced as ordered polylines.
"""
import json, numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

FIGDIR="figures"; OUT="extracted"
import os; os.makedirs(OUT,exist_ok=True)

def load(fn):
    return np.asarray(Image.open(os.path.join(FIGDIR,fn)).convert("RGB")).astype(int)

def mask_color(img, rgb, tol):
    d=np.abs(img-np.array(rgb)).sum(axis=2)
    return d<tol

def lin(px, p0,v0, p1,v1):
    return v0+(px-p0)*(v1-v0)/(p1-p0)

def clusters(mask, min_size=6):
    lab,n=ndimage.label(mask)
    out=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        if len(xs)>=min_size:
            out.append((xs.mean(), ys.mean(), len(xs), xs.min(),xs.max(),ys.min(),ys.max()))
    return out

results={}

# ---------- A) B3 RM growth factor : magenta filled circles ----------
def figA():
    fn="knapp_rm_growth_rate.png"; img=load(fn)
    # calib (px->data), linear both
    X=lambda px: lin(px,357.5,0.0,1971.7,10.0)
    Y=lambda py: lin(py,295.9,4.5,1461.0,1.0)
    m=mask_color(img,(255,0,255),120)
    cl=clusters(m,min_size=20)
    pts=sorted([(X(cx),Y(cy)) for cx,cy,*_ in cl])
    # drop spurious legend swatch: legend usually upper-right; keep physically sane (gf>=0.8)
    pts=[(x,y) for x,y in pts if -0.5<x<12 and 0.7<y<5.2]
    results["B3_rm_growth_factor"]={"figure":fn,"x":"k_z*delta_r (dimensionless)","y":"growth factor h/h0 (dimensionless)","experiment_color":"magenta","points":[{"x":round(x,3),"y":round(y,3)} for x,y in pts]}
    # overlay
    fig,ax=plt.subplots(figsize=(7,6)); ax.imshow(Image.open(os.path.join(FIGDIR,fn)))
    px=[357.5+(x-0)*(1971.7-357.5)/10 for x,y in pts]; py=[295.9+(y-4.5)*(1461.0-295.9)/(1.0-4.5) for x,y in pts]
    ax.scatter(px,py,s=160,facecolors='none',edgecolors='lime',linewidths=2.0,label='extracted exp pts')
    ax.set_title("B3 RM growth — extracted (lime) over paper magenta 'Z Data'"); ax.legend(loc='lower right')
    fig.savefig(os.path.join(OUT,"B3_rm_growth_overlay.png"),dpi=110,bbox_inches='tight'); plt.close(fig)

# ---------- D) B1 single-mode MRT amplitude : blue '+' (log y, mm) ----------
def figD():
    fn="sinars_mrt_growth_rate.png"; img=load(fn)
    X=lambda px: lin(px,99.5,0.0,530.0,60.0)
    # log y: px28.5->10^0, px358.5->10^-2  => log10 linear
    def Y(py): return 10**lin(py,28.5,0.0,358.5,-2.0)
    # pure blue + (exclude lavender squares & dashed): require strong blue, low red/green
    r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
    m=(b>150)&(r<90)&(g<90)
    cl=clusters(m,min_size=8)
    pts=sorted([(X(cx),Y(cy)) for cx,cy,*_ in cl])
    pts=[(x,y) for x,y in pts if -2<x<90 and 1e-3<y<3]
    results["B1_single_mode_amplitude"]={"figure":fn,"x":"time (ns)","y":"mode amplitude (mm, log)","experiment_color":"blue +","points":[{"t_ns":round(x,2),"amp_mm":round(y,4)} for x,y in pts]}
    fig,ax=plt.subplots(figsize=(7,5)); ax.imshow(Image.open(os.path.join(FIGDIR,fn)))
    px=[99.5+(x)*(530-99.5)/60 for x,y in pts]; py=[28.5+(np.log10(y)-0)*(358.5-28.5)/(-2.0) for x,y in pts]
    ax.scatter(px,py,s=120,facecolors='none',edgecolors='lime',linewidths=1.8)
    ax.set_title("B1 MRT amplitude — extracted (lime) over paper blue '+' Experiment")
    fig.savefig(os.path.join(OUT,"B1_amplitude_overlay.png"),dpi=110,bbox_inches='tight'); plt.close(fig)

# ---------- E) B5 z2977 BR : red filled circles ----------
def figE():
    fn="z2977_rb.png"; img=load(fn)
    X=lambda px: lin(px,74.5,0.25,512.4,1.75)
    Y=lambda py: lin(py,22.4,0.6,323.0,0.1)
    r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
    m=(r>150)&(g<90)&(b<90)
    cl=clusters(m,min_size=12)
    pts=sorted([(X(cx),Y(cy)) for cx,cy,*_ in cl])
    pts=[(x,y) for x,y in pts if 0.0<x<2.2 and 0.05<y<0.7]
    results["B5_BR_vs_preheat"]={"figure":fn,"x":"laser preheat energy (kJ)","y":"BR (MG*cm)","experiment_color":"red circles ('Data')","points":[{"preheat_kJ":round(x,3),"BR_MGcm":round(y,3)} for x,y in pts]}
    fig,ax=plt.subplots(figsize=(7,5)); ax.imshow(Image.open(os.path.join(FIGDIR,fn)))
    px=[74.5+(x-0.25)*(512.4-74.5)/(1.75-0.25) for x,y in pts]; py=[22.4+(y-0.6)*(323-22.4)/(0.1-0.6) for x,y in pts]
    ax.scatter(px,py,s=160,facecolors='none',edgecolors='lime',linewidths=2.0)
    ax.set_title("B5 BR — extracted (lime) over paper red 'Data'")
    fig.savefig(os.path.join(OUT,"B5_BR_overlay.png"),dpi=110,bbox_inches='tight'); plt.close(fig)

# ---------- B) B3 trajectory : blue solid (liner) + green solid (shock) lines ----------
def figB():
    fn="knapp_rm_r_vs_t.png"; img=load(fn)
    X=lambda px: lin(px,83.4,2850.0,626.2,3150.0)
    Y=lambda py: lin(py,44.5,5.0,276.7,1.0)
    r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
    def trace(mask):
        cols={}
        ys,xs=np.where(mask)
        for x,y in zip(xs,ys): cols.setdefault(x,[]).append(y)
        return sorted((X(x),Y(np.mean(v))) for x,v in cols.items() if len(v)>=1)
    liner=(b>120)&(r<90)&(g<120)        # blue
    shock=(g>110)&(r<120)&(b<120)       # green
    lt=trace(liner); st=trace(shock)
    # downsample to ~25 pts each
    def ds(p,n=25):
        if len(p)<=n: return p
        idx=np.linspace(0,len(p)-1,n).astype(int); return [p[i] for i in idx]
    results["B3_trajectory_liner_exp"]={"figure":fn,"x":"time (ns)","y":"inner liner radius (mm)","series":"PDV experiment (blue)","points":[{"t_ns":round(x,1),"r_mm":round(y,3)} for x,y in ds(lt)]}
    results["B3_trajectory_shock_exp"]={"figure":fn,"x":"time (ns)","y":"shock radius (mm)","series":"PDV experiment (green)","points":[{"t_ns":round(x,1),"r_mm":round(y,3)} for x,y in ds(st)]}
    fig,ax=plt.subplots(figsize=(7,5)); ax.imshow(Image.open(os.path.join(FIGDIR,fn)))
    for p,c in [(ds(lt),'cyan'),(ds(st),'lime')]:
        px=[83.4+(x-2850)*(626.2-83.4)/300 for x,y in p]; py=[44.5+(y-5)*(276.7-44.5)/(1-5) for x,y in p]
        ax.plot(px,py,'.',color=c,ms=4)
    ax.set_title("B3 trajectory — extracted liner(cyan)/shock(lime) over paper PDV")
    fig.savefig(os.path.join(OUT,"B3_trajectory_overlay.png"),dpi=110,bbox_inches='tight'); plt.close(fig)

# ---------- C) B4 confinement panel(b): gray open circles w/ error bars ----------
def figC():
    fn="knapp_confine_r_vs_t.png"; img=load(fn)
    # panel b only: x in [833,1419]
    X=lambda px: lin(px,833.1,3115.0,1419.0,3140.0)
    Y=lambda py: lin(py,37.6,0.65,422.4,0.30)
    r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
    grayish=(np.abs(r-g)<25)&(np.abs(g-b)<25)&(r>70)&(r<180)
    panel=np.zeros_like(grayish); panel[:,833:1420]=True
    m=grayish&panel
    cl=clusters(m,min_size=10)
    pts=sorted([(X(cx),Y(cy),y0,y1) for cx,cy,sz,x0,x1,y0,y1 in cl])
    out=[{"t_ns":round(x,1),"r_mm":round(y,3),"err_lo_mm":round(Y(yb1),3),"err_hi_mm":round(Y(yb0),3)} for x,y,yb0,yb1 in pts if 3113<x<3142 and 0.30<y<0.66]
    results["B4_inner_radius_points_exp"]={"figure":fn+" (panel b)","x":"time (ns)","y":"inner liner radius (mm)","series":"radiograph experiment (gray circles w/ errbars)","confidence":"LOW - overlapping FLASH markers contaminate; verify by eye","points":out}
    fig,ax=plt.subplots(figsize=(8,4)); ax.imshow(Image.open(os.path.join(FIGDIR,fn)))
    px=[833.1+(d["t_ns"]-3115)*(1419-833.1)/25 for d in out]; py=[37.6+(d["r_mm"]-0.65)*(422.4-37.6)/(0.30-0.65) for d in out]
    ax.scatter(px,py,s=120,facecolors='none',edgecolors='red',linewidths=1.8)
    ax.set_title("B4 inner-radius — extracted (red) over paper panel(b) [LOW confidence]")
    fig.savefig(os.path.join(OUT,"B4_inner_radius_overlay.png"),dpi=110,bbox_inches='tight'); plt.close(fig)

for f in (figA,figD,figE,figB,figC):
    try: f()
    except Exception as e: print("ERR",f.__name__,repr(e))

json.dump(results,open(os.path.join(OUT,"extracted_points.json"),"w"),indent=2)
print("WROTE",os.path.join(OUT,"extracted_points.json"))
for k,v in results.items():
    n=len(v.get("points",[])); print(f"  {k}: {n} pts")
