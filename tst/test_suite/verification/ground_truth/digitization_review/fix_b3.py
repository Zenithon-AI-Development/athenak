import numpy as np, json, os
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def load(fn): return np.asarray(Image.open(os.path.join("figures",fn)).convert("RGB")).astype(int)
fn="knapp_rm_r_vs_t.png"; img=load(fn); H,W,_=img.shape
r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
Xt=lambda px:2850.0+(px-83.4)*(3150.0-2850.0)/(626.2-83.4)
Yr=lambda py:5.0+(py-44.5)*(1.0-5.0)/(276.7-44.5)
Tinv=lambda t:83.4+(t-2850)*(626.2-83.4)/300.0
Rinv=lambda rr:44.5+(rr-5)*(276.7-44.5)/(1-5)

def groups_in_col(ys, gap=4):
    ys=sorted(ys); out=[]; cur=[ys[0]]
    for y in ys[1:]:
        if y-cur[-1]<=gap: cur.append(y)
        else: out.append(cur); cur=[y]
    out.append(cur); return [np.median(c) for c in out]

def trace(mask, start='top', y_tol=3.0):
    cols={}; ys,xs=np.where(mask)
    for x,y in zip(xs,ys): cols.setdefault(x,[]).append(y)
    xs_sorted=[x for x in sorted(cols) if len(cols[x])>=2]
    pts=[]; prev=None
    for x in xs_sorted:
        gs=groups_in_col(cols[x])
        if prev is None:
            cy=min(gs) if start=='top' else max(gs)  # top = smallest py = largest r
        else:
            # monotonic descent: r decreases => py increases. require py >= prev - y_tol; pick closest
            cand=[gy for gy in gs if gy >= prev - y_tol]
            if not cand: continue
            cy=min(cand, key=lambda gy: abs(gy-prev))
        prev=cy; pts.append((Xt(x), Yr(cy)))
    return pts

# liner (blue) - good already; trace topmost continuity (liner also monotonic decreasing)
blue=(b>120)&(r<90)&(g<120)
liner=trace(blue, start='top')
# shock (green) strict mask + continuity/monotonic
green=(g>140)&(r<90)&(b<110)&(g-r>50)&(g-b>40)
shock=trace(green, start='top')

def ds(p,n=30):
    if len(p)<=n: return p
    idx=np.linspace(0,len(p)-1,n).astype(int); return [p[i] for i in idx]
liner=ds(liner); shock=ds(shock)
print(f"liner {len(liner)} pts  r[{min(p[1] for p in liner):.2f},{max(p[1] for p in liner):.2f}] t[{liner[0][0]:.0f},{liner[-1][0]:.0f}]")
print(f"shock {len(shock)} pts  r[{min(p[1] for p in shock):.2f},{max(p[1] for p in shock):.2f}] t[{shock[0][0]:.0f},{shock[-1][0]:.0f}]")
# monotonicity check
def mono(p): 
    bad=[(round(p[i][0]),round(p[i][1],2)) for i in range(1,len(p)) if p[i][1] > p[i-1][1]+0.2]
    return bad
print("shock non-monotonic jumps:", mono(shock))

# overlay
fig,ax=plt.subplots(figsize=(8,6)); ax.imshow(Image.open(os.path.join("figures",fn)))
for pts,c,lab in [(liner,'cyan','liner'),(shock,'lime','shock')]:
    px=[Tinv(t) for t,_ in pts]; py=[Rinv(rr) for _,rr in pts]
    ax.plot(px,py,'o',color=c,ms=5,mec='black',mew=0.4,label=f"{lab} ({len(pts)})")
ax.legend(loc='upper right'); ax.set_title("B3 trajectory v3 — continuity+monotonic trace")
fig.savefig("extracted/B3_trajectory_overlay_v3.png",dpi=120,bbox_inches='tight'); plt.close(fig)
json.dump({"B3_trajectory_liner_exp":{"series":"PDV liner (blue)","qc":"GOOD","points":[{"t_ns":round(t,1),"r_mm":round(rr,3)} for t,rr in liner]},
           "B3_trajectory_shock_exp":{"series":"PDV shock (green)","qc":"GOOD - continuity+monotonic trace; spurious upper strand culled","points":[{"t_ns":round(t,1),"r_mm":round(rr,3)} for t,rr in shock]}},
          open("extracted/b3_trajectory_v3.json","w"),indent=2)
print("wrote B3_trajectory_overlay_v3.png + b3_trajectory_v3.json")
