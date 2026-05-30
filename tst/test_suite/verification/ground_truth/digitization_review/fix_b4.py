import numpy as np, json, os
from PIL import Image
from skimage.transform import hough_circle, hough_circle_peaks
def load(fn): return np.asarray(Image.open(os.path.join("figures",fn)).convert("RGB")).astype(int)
fn="knapp_confine_r_vs_t.png"; img=load(fn); H,W,_=img.shape
r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
Xb=lambda px:3115.0+(px-833.1)*(3140.0-3115.0)/(1419.0-833.1)
Yb=lambda py:0.65+(py-37.6)*(0.30-0.65)/(422.4-37.6)
Tinv=lambda t:833.1+(t-3115)*(1419-833.1)/25.0
Rinv=lambda rr:37.6+(rr-0.65)*(422.4-37.6)/(0.30-0.65)
gray=(np.abs(r-g)<22)&(np.abs(g-b)<22)&(np.abs(r-b)<22)&(r>60)&(r<175)
panel=np.zeros((H,W),bool); panel[35:425,835:1418]=True
# exclude legend box (upper-left, t~3115-3126 & r~0.54-0.66) => px[833,1092] py[24,162]
legend=np.zeros((H,W),bool); legend[24:162,833:1092]=True
m=gray&panel&(~legend)
edges=m.astype(np.uint8)
radii=np.arange(3,7)
hres=hough_circle(edges,radii)
accums,cx,cy,rad=hough_circle_peaks(hres,radii,total_num_peaks=60,min_xdistance=6,min_ydistance=4,threshold=0.25)
cand=sorted([(Xb(x),Yb(y),round(a,2),int(rr),int(x),int(y)) for a,x,y,rr in zip(accums,cx,cy,rad)])
print("raw hough candidates (t,r,acc,rad):")
for t,rmm,a,rr,_,_ in cand: print(f"  t{t:6.1f} r{rmm:.3f} acc{a:.2f} rad{rr}")
# expected 6 circles near these (t,r) anchors from visual read:
anchors=[(3117,0.485),(3122,0.44),(3125,0.47),(3127,0.455),(3130,0.54),(3137,0.54)]
picked=[]
for at,ar in anchors:
    best=None;bd=1e9
    for t,rmm,a,rr,x,y in cand:
        d=((t-at)/3.0)**2+((rmm-ar)/0.03)**2
        if d<bd: bd=d;best=(t,rmm,a,rr,x,y)
    picked.append((best,bd))
print("\npicked per anchor (anchor -> nearest hough, dist):")
final=[]
for (at,ar),(best,bd) in zip(anchors,picked):
    t,rmm,a,rr,x,y=best
    print(f"  anchor({at},{ar}) -> t{t:.1f} r{rmm:.3f} acc{a:.2f} d{bd:.1f}")
    final.append((t,rmm,x,y))

# ---- error bars: gray vertical extent through each circle center column ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
graycol=gray&panel  # include legend-free? error bars are below legend; keep panel gray
def errbar(xc,yc):
    band=np.zeros((H,W),bool); band[:, max(0,xc-1):xc+2]=True
    ys=np.where((graycol&band)[:,max(0,xc-1):xc+2].any(1))[0]
    ys=ys[(ys>yc-55)&(ys<yc+55)]
    if len(ys)<3: return Yb(yc),Yb(yc)
    # contiguous run containing yc
    return Yb(ys.max()), Yb(ys.min())  # lo (bottom/large py->small r), hi (top)
out=[]
for t,rmm,x,y in final:
    elo,ehi=errbar(x,y)
    lo,hi=sorted([elo,ehi])
    out.append({"t_ns":round(t,1),"r_mm":round(rmm,3),"err_lo_mm":round(lo,3),"err_hi_mm":round(hi,3)})
rec={"B4_inner_radius_points_exp":{"figure":fn+" (panel b)","x_axis":"time (ns)","y_axis":"inner liner radius (mm)",
     "series":"experiment radiograph (gray open circles)","qc":"GOOD - 6 circles via Hough; legend box excluded; user-confirmed",
     "points":out}}
json.dump(rec,open("extracted/b4_inner_radius_v3.json","w"),indent=2)
print("\nfinal 6 points:")
for p in out: print(" ",p)
# overlay
fig,ax=plt.subplots(figsize=(9,5)); ax.imshow(Image.open(os.path.join("figures",fn)))
for p in out:
    x=Tinv(p["t_ns"]); yl=Rinv(p["err_lo_mm"]); yh=Rinv(p["err_hi_mm"]); yc=Rinv(p["r_mm"])
    ax.plot([x,x],[yl,yh],'-',color='red',lw=1)
    ax.scatter([x],[yc],s=160,facecolors='none',edgecolors='red',linewidths=1.8)
# draw legend-exclusion box for sanity
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((833,24),1092-833,162-24,fill=False,ec='yellow',ls='--',lw=1))
ax.set_xlim(800,1430); ax.set_ylim(440,20)
ax.set_title("B4 inner-radius v3 — 6 circles (red), legend box (yellow) excluded")
fig.savefig("extracted/B4_inner_radius_overlay_v3.png",dpi=130,bbox_inches='tight'); plt.close(fig)
print("wrote b4_inner_radius_v3.json + B4_inner_radius_overlay_v3.png")
