import sys, subprocess, json
sys.path.insert(0,'/mnt/project-files/football-analysis')
import cv2, numpy as np
from football_analysis.body_events.synthetic import generate, seeds_for
from football_analysis.body_events.detector import BodyEventConfig, detect_body_events
from football_analysis.body_events.model import BodyEventModel, default_model_path
FF='/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
OUT='/mnt/project-files/football-analysis/assets/body_events/examples'
EDGES=[(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]
COL=[(80,200,255),(255,140,60)]
m=BodyEventModel.load(default_model_path()); cfg=BodyEventConfig()

def frame(s, c, H, W=960, Hh=540, sc=None):
    im=np.full((Hh,W,3),(40,90,40),np.uint8)
    sc=sc or Hh/(3.2*H)
    f=lambda x,y:(int((x-c[0])*sc+W/2),int((y-c[1])*sc+Hh/2))
    for pi,p in enumerate(sorted(s.players,key=lambda p:p.player_id)):
        col=COL[pi%2]
        if len(p.keypoints)==17:
            for a,b in EDGES:
                A,B=p.keypoints[a],p.keypoints[b]
                if A.confidence>0.3 and B.confidence>0.3:
                    cv2.line(im,f(A.x,A.y),f(B.x,B.y),col,3,cv2.LINE_AA)
        cv2.putText(im,p.player_id,f(p.bbox.x1,p.bbox.y1-4),0,0.6,col,2,cv2.LINE_AA)
    if s.ball:
        cv2.circle(im,f(s.ball.position.x,s.ball.position.y),max(4,int(0.06*H*sc)),(255,255,255) if not s.ball.interpolated else (150,150,150),-1,cv2.LINE_AA)
    return im

def render(seq, name, truth):
    ev=sorted(detect_body_events(seq.states,m,cfg),key=lambda e:e.timestamp_s)
    st=seq.states
    H=float(np.median([p.bbox.height for s in st for p in s.players]))
    cx=np.median([p.bbox.center.x for s in st for p in s.players]); cy=np.median([p.bbox.center.y for s in st for p in s.players])
    mids=np.array([[np.mean([p.bbox.center.x for p in f.players]),np.mean([p.bbox.center.y for p in f.players])] for f in st])
    k=np.ones(25)/25; pad=np.pad(mids,((12,12),(0,0)),mode='edge')
    cam=np.stack([np.convolve(pad[:,j],k,mode='valid') for j in range(2)],1)
    raw=f'/tmp/claude-0/work/{name}.avi'
    wr=cv2.VideoWriter(raw,cv2.VideoWriter_fourcc(*'MJPG'),25,(960,540))
    sheet=[]
    for s in st:
        im=frame(s,cam[st.index(s)],H)
        cv2.putText(im,f"{s.timestamp_s:5.2f}s   truth: {truth}",(12,30),0,0.7,(230,230,230),2,cv2.LINE_AA)
        live=[e for e in ev if e.timestamp_s-0.05<=s.timestamp_s<=max(e.end_timestamp_s or e.timestamp_s, e.timestamp_s)+1.0]
        y=64
        for e in live:
            d=e.detail
            txt=f"CALL {e.type.value.upper()}  {d.get('outcome') or d.get('trick_name')}  conf {e.confidence:.2f}  by {e.player_id}"
            cv2.putText(im,txt,(12,y),0,0.8,(0,255,255),2,cv2.LINE_AA); y+=32
        if not ev:
            cv2.putText(im,"no call",(12,64),0,0.8,(180,180,180),2,cv2.LINE_AA)
        wr.write(im)
    wr.release()
    mp4=f'{OUT}/{name}.mp4'
    subprocess.run([FF,'-y','-loglevel','error','-i',raw,'-c:v','libx264','-pix_fmt','yuv420p','-crf','23','-movflags','+faststart',mp4],check=True)
    # sheet: 6 frames across the event
    ts=np.linspace(max(0,seq.event_start_s-0.4),min(st[-1].timestamp_s,seq.event_end_s+0.6),6)
    cells=[]
    for t in ts:
        i=int(np.argmin([abs(s.timestamp_s-t) for s in st])); im=frame(st[i],cam[i],H)
        cv2.putText(im,f"{st[i].timestamp_s:.2f}s",(12,30),0,0.8,(230,230,230),2)
        for e in ev:
            if abs(e.timestamp_s-st[i].timestamp_s)<0.25:
                cv2.putText(im,"call",(12,64),0,0.8,(0,255,255),2)
        cells.append(cv2.resize(im,(480,270)))
    img=np.vstack([np.hstack(cells[:3]),np.hstack(cells[3:])])
    cv2.imwrite(f'{OUT}/{name}.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,88])
    return [dict(type=e.type.value,t=round(e.timestamp_s,2),conf=round(e.confidence,2),name=e.detail.get('outcome') or e.detail.get('trick_name')) for e in ev]

want={'tackle_won':'tackle, defender wins it','stepover':'stepover (trick)','nutmeg':'nutmeg (trick)','shoulder_duel':'shoulder bump, not a tackle','turn':'ordinary turn, not a trick'}
seeds={n:s for n,s in seeds_for(40,7) }
res={}
for n,label in want.items():
    for n2,s in seeds_for(40,7):
        if n2!=n: continue
        seq=generate(n,s)
        ev=detect_body_events(seq.states,m,cfg)
        ok = (n in ('shoulder_duel','turn') and not ev) or (n=='tackle_won' and any(e.type.value=='tackle' for e in ev)) or (n=='stepover' and any(e.type.value=='trick' for e in ev)) or (n=='nutmeg' and any(e.detail.get('trick_name')=='nutmeg' for e in ev))
        if ok:
            res[n]=dict(seed=s,events=render(seq,f'sim_{n}',label)); break
print(json.dumps(res,indent=1))
