"""Figures added for the restructured final report (v2); the shared ones come from figures.py.

Run with the repository venv (the SM timeline imports kernelscope): .venv/bin/python docs/report/final/src/figures_v2.py
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT=Path(__file__).resolve().parent.parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont('/root/.local/share/fonts/malgun.ttf')
font_manager.fontManager.addfont('/root/.local/share/fonts/malgunbd.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
C=['#284B63','#4C8F95','#BD7339','#7B6E92']
NAVY,TEAL,ORANGE,GRAY='#284B63','#4C8F95','#BD7339','#8A8F94'
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),facecolor='white',bbox_inches='tight',pad_inches=.12)
 fig.savefig(OUT/(name+'.svg'),facecolor='white',bbox_inches='tight',pad_inches=.12)
 plt.close(fig)

# Box styles: new = built in this work, ext = existing tool extended, old = used as is, warn = the problem.
STYLE={'new':dict(fc=NAVY,ec=NAVY,lw=1.0,tc='white'),'ext':dict(fc='white',ec=TEAL,lw=2.0,tc='#1D2B33'),
 'old':dict(fc='#ECEEEF',ec=GRAY,lw=1.0,tc='#1D2B33'),'warn':dict(fc='#FBEBDD',ec=ORANGE,lw=1.6,tc='#1D2B33'),
 'soft':dict(fc='#E3EEF0',ec=TEAL,lw=1.0,tc='#1D2B33')}
def box(ax,x,y,w,h,title,sub='',style='old',ts=8.6,ss=7.0,pad=.35):
 s=STYLE[style]
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad={pad}',fc=s['fc'],ec=s['ec'],lw=s['lw'],zorder=2))
 if sub:
  ax.text(x+w/2,y+h*.68,title,ha='center',va='center',fontsize=ts,fontweight='bold',color=s['tc'],zorder=3)
  ax.text(x+w/2,y+h*.29,sub,ha='center',va='center',fontsize=ss,color=s['tc'],zorder=3,linespacing=1.15)
 else:
  ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=ts,fontweight='bold',color=s['tc'],zorder=3,linespacing=1.15)
def arrow(ax,a,b,color=NAVY,ls='-',lw=1.3,text=None,tpos=.5,toff=(0,1.1),ts=6.8,rad=0):
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=color,lw=lw,ls=ls,mutation_scale=9,
  shrinkA=0,shrinkB=0,connectionstyle=f'arc3,rad={rad}'),zorder=4)
 if text:
  ax.text(a[0]+(b[0]-a[0])*tpos+toff[0],a[1]+(b[1]-a[1])*tpos+toff[1],text,ha='center',va='center',fontsize=ts,color=color,zorder=5,
   bbox=dict(fc='white',ec='none',pad=.4))
def badge(ax,x,y,n,color=ORANGE,size=8.5):
 ax.text(x,y,n,ha='center',va='center',fontsize=size,fontweight='bold',color='white',zorder=6,
  bbox=dict(boxstyle='circle,pad=0.18',fc=color,ec='none'))

# ---------------------------------------------------------------- CTA timeline from the calibrated simulator
from kernelscope.workload import Workload
from kernelscope.model.machine import MachineSpec
from kernelscope.model.params import ModelParams
from kernelscope.model.predict import bandwidth_bytes_per_us
from kernelscope.model.geometry import build_launch, parse_variant, KIND_RESOURCES
from kernelscope.backends.realhw.kprofile import blocks_per_sm_limit
from kernelscope.analytic import DTYPE_BYTES
MACHINE=MachineSpec.from_json(REPO/'machines/rtx4090.json')
PARAMS=ModelParams.from_json(REPO/'models/rtx4090.json')
RAGGED=(32768,)+(512,)*31
W=Workload('decode',32,1,32768,32,8,128,'float16',True,kv_lens=RAGGED)


def cta_timeline(variant,cache='cold'):
 """The reference event loop of kernelscope/model/simulate.py, also recording each CTA's SM, start and end (µs)."""
 m,p=MACHINE,PARAMS
 launch=build_launch(W,parse_variant(variant),m.n_sm);kp=p.states[cache].kinds[launch.kind]
 slots,_=blocks_per_sm_limit(*KIND_RESOURCES[launch.kind],m.props())
 bpk=2*W.d*DTYPE_BYTES[W.dtype];bw=bandwidth_bytes_per_us(m,cache,float(launch.keys.sum())*bpk)
 keys=launch.keys.astype(float);n=len(keys);ns=m.n_sm*slots
 slot_sm=np.asarray(m.sm_order())[np.arange(ns)%m.n_sm]
 occ=np.zeros(ns,bool);rf=np.zeros(ns);rk=np.zeros(ns);cta=np.full(ns,-1)
 sm,start,end=np.zeros(n,int),np.zeros(n),np.zeros(n);nxt=0;t=0.0
 def admit(s):
  nonlocal nxt
  k=min(len(s),n-nxt)
  if k<=0:return
  s=s[:k];ids=np.arange(nxt,nxt+k);nxt+=k
  occ[s]=True;rk[s]=keys[ids];rf[s]=np.where(keys[ids]>0,kp.t0_us,kp.t_empty_us);cta[s]=ids;sm[ids]=slot_sm[s];start[ids]=t
 admit(np.arange(ns))
 while occ.any():
  per=np.bincount(slot_sm[occ],minlength=m.n_sm);g=np.where(per[slot_sm]>=2,kp.gamma,1.0)
  fx=occ&(rf>1e-9);st=occ&~fx&(rk>1e-6);rate=np.where(st,g/kp.cost_us_per_key,0.0)
  d=rate.sum()*bpk;rate*=1.0 if d<=bw else bw/d
  dt=0.0 if (occ&~fx&~st).any() else min(np.min(rf[fx]) if fx.any() else np.inf,np.min(rk[st]/rate[st]) if st.any() else np.inf)
  t+=dt;rf[fx]-=dt;rk[st]-=rate[st]*dt
  fin=occ&(rf<=1e-9)&(rk<=1e-6);end[cta[fin]]=t;occ[fin]=False;admit(np.flatnonzero(fin))
 return sm,start,end


def busy_per_sm(variant):
 sm,_,end=cta_timeline(variant)
 out=np.zeros(MACHINE.n_sm)
 np.maximum.at(out,sm,end)
 return out


def active_sms(variant,grid):
 sm,start,end=cta_timeline(variant)
 return np.array([len(set(sm[(start<=x)&(end>x)])) for x in grid])

# ---------------------------------------------------------------- Figure: the problem, default path vs this work
fig,ax=plt.subplots(figsize=(7.7,3.4));ax.set(xlim=(0,100),ylim=(0,44));ax.axis('off')
rows=[(23.2,'기존',ORANGE,('기본 휴리스틱','배치 크기·최대 길이만 봄\n→ 분할 1 (나누지 않음)','warn'),busy_per_sm('fa2'),
       '커널 3,501 µs (실측)'),
      (1.2,'제안',NAVY,('KernelScope 선택기','요청별 길이로 후보 예측\n→ 분할 16','new'),busy_per_sm('fd_s16'),
       '커널 279 µs (실측), 같은 출력')]
ymax=max(r[4].max() for r in rows)
for y0,label,col,(dt,ds,dstyle),busy,res in rows:
 ax.add_patch(FancyBboxPatch((0.4,y0),99.2,19.2,boxstyle='round,pad=0.2',fc='#F4F7F9' if label=='제안' else '#FDF7F1',ec='none',zorder=0))
 ax.text(3.0,y0+9.6,label,ha='center',va='center',fontsize=10.5,fontweight='bold',color=col,rotation=90)
 box(ax,7,y0+2.4,17,14.4,'','','old')
 ax.text(15.5,y0+14.6,'디코드 배치 (B=32)',ha='center',fontsize=7.8,fontweight='bold')
 for i,n in enumerate(RAGGED):
  h=8.0 if i==0 else 0.9
  ax.add_patch(Rectangle((8.3+i*0.455,y0+5.0),0.32,h,fc=ORANGE if i==0 else GRAY,ec='none',zorder=3))
 ax.text(15.5,y0+3.4,'32K×1 + 512×31',ha='center',fontsize=7)
 box(ax,29,y0+3.8,19,11.4,dt,ds,dstyle,ts=8.6,ss=7.0)
 box(ax,53,y0+3.8,14,11.4,'FlashAttention-2','커널·가중치\n그대로','old',ts=8.0,ss=7.0)
 for xa,xb in [(24.6,28.4),(48.6,52.4),(67.6,71.4)]:arrow(ax,(xa,y0+9.5),(xb,y0+9.5))
 bx,bw_,by,bh=72.6,26.4,y0+5.0,10.6
 ax.add_patch(Rectangle((bx,by),bw_,bh,fc='white',ec='#C9CED2',lw=.8,zorder=1))
 for i,v in enumerate(busy):ax.add_patch(Rectangle((bx+.3+i*(bw_-.6)/128,by),(bw_-.6)/128*.85,max(v/ymax*bh,.35),fc=col,ec='none',zorder=2))
 ax.text(bx+bw_/2,y0+17.4,'SM 128개가 각각 일한 시간',ha='center',fontsize=7.4,color='#333333')
 ax.text(bx+bw_/2,y0+2.6,res,ha='center',va='center',fontsize=7.2,fontweight='bold',color=col)
ax.text(72.6+26.4-.6,23.2+5.0+10.6-1.0,'↑ 3.5 ms',fontsize=6.3,color='#666666',va='center',ha='right')
ax.text(72.6+14.0,23.2+5.0+6.0,'긴 요청의 CTA가 몰린 SM',fontsize=6.6,color=ORANGE,va='center',zorder=5,bbox=dict(fc='white',ec='none',pad=.2,alpha=.9))
ax.text(72.6+1.0,1.2+5.0+5.6,'모든 SM이 고르게, 짧게 일함',fontsize=6.6,color=NAVY,va='center',zorder=5)
save(fig,'concept')

# ---------------------------------------------------------------- Figure: where existing analysis methods sit
fig,ax=plt.subplots(figsize=(7.7,3.0))
ax.set_xscale('log');ax.set_xlim(3e-7,1e4);ax.set_ylim(-.65,1.75)
ax.axvspan(3e-7,0.03425,ymin=(0.55+.65)/2.4,ymax=1,color='#E3EEF0',zorder=0)
ax.text(5e-7,1.6,'매 단계 선택에 쓸 수 있고 요청별 길이를 보는 영역',fontsize=8,color=TEAL,va='center')
ax.axvline(0.03425,color=GRAY,ls='--',lw=1,zorder=1);ax.text(0.03425*1.25,-.52,'디코드 1단계(34 ms)',fontsize=8,color='#555555')
ax.axhline(.5,color='#DDDDDD',lw=.8,zorder=0)
ax.scatter([1e-6],[0],s=70,color=GRAY,zorder=3);ax.text(1.4e-6,0.13,'FA2 기본 휴리스틱\n(B, 최대 길이만)',fontsize=8,va='bottom')
ax.scatter([1e-5],[0],s=70,color=GRAY,zorder=3);ax.text(1.4e-5,-0.12,'루프라인 모델\n(평균 작업량)',fontsize=8,va='top')
ax.plot([19.06,992.58],[1,1],color=ORANGE,lw=7,solid_capstyle='round',zorder=3)
ax.text(19.06*0.6,1.17,'Accel-Sim 19 s~17 min\n(공개판은 RTX 4090 미지원)',fontsize=8,va='bottom',color=ORANGE)
ax.scatter([330],[.8],s=70,color='#D9A066',zorder=3);ax.text(330*1.4,.8,'GPGPU-Sim\n330 s',fontsize=8,va='center',color='#B07A3A')
ax.scatter([0.8426],[1],s=60,facecolors='white',edgecolors=C[3],lw=1.5,zorder=3);ax.text(0.8426,.8,'초기 NumPy\n0.84 s',fontsize=7.5,ha='center',va='top',color=C[3])
ax.scatter([0.014293],[1],s=230,marker='*',color=NAVY,zorder=4)
ax.text(0.014293*0.7,1.17,'본 작품 CTA 시뮬레이터\n첫 선택 14.3 ms',fontsize=8.3,fontweight='bold',ha='right',va='bottom',color=NAVY)
ax.scatter([5.567e-6],[1],s=50,marker='*',color=TEAL,zorder=4);ax.text(5.567e-6,.83,'결정 캐시 적중 µs',fontsize=7.5,ha='center',va='top',color=TEAL)
ax.set_yticks([0,1],labels=['대표값으로 요약\n(배치 크기·평균·최대 길이)','요청별 길이 반영\n(CTA 단위)'],fontsize=8.5)
ax.set_xticks([1e-6,1e-4,1e-2,1,1e2,1e4],labels=['1 µs','100 µs','10 ms','1 s','100 s','10,000 s'],fontsize=8.5)
ax.set_xlabel('한 번 평가하는 데 걸리는 시간 (로그 축)',fontsize=9);ax.tick_params(axis='y',length=0)
ax.spines['left'].set_visible(False);ax.grid(axis='x',alpha=.15)
fig.tight_layout();save(fig,'method_map')

# ---------------------------------------------------------------- Figure: the framework (offline analysis, online selection, verification)
fig,ax=plt.subplots(figsize=(7.7,4.85));ax.set(xlim=(0,100),ylim=(-4.2,63));ax.axis('off')
def band(y,h,title,col):
 ax.add_patch(FancyBboxPatch((0.5,y),99,h,boxstyle='round,pad=0.3',fc='#F5F8F9',ec='#C9D6DB',lw=.9,zorder=0))
 ax.text(2,y+h-1.7,title,fontsize=8.8,fontweight='bold',color=col,va='center')
def tag(x,y,t):
 ax.text(x,y,t,ha='left',va='center',fontsize=7.2,fontweight='bold',color='white',zorder=6,
  bbox=dict(boxstyle='round,pad=0.25',fc=ORANGE,ec='none'))
band(33.0,29.4,'분석 환경 (오프라인: RTX 4090 측정 + CPU 분석)',NAVY);tag(50.2,60.7,'한계 ① 해결')
band(12.2,18.8,'KernelScope 선택기 (온라인: 매 디코드 단계)',NAVY);tag(47.0,29.3,'한계 ② 해결')
band(0.4,10.0,'검증·재현',NAVY)
# inputs
ax.text(10.0,56.0,'입력',ha='center',fontsize=7.6,color='#555555')
for i,(t,s_) in enumerate([('RTX 4090','SM 128 · L2 72 MiB'),('모델 헤드 구성','Qwen3-4B · Llama 8B'),('공개 트레이스','Azure · BurstGPT · Mooncake')]):
 box(ax,2.4,48.8-i*7.2,15.2,5.2,t,s_,'old',ts=7.6,ss=6.0,pad=.3)
# measurement layer
ax.text(29.0,56.0,'측정 계층',ha='center',fontsize=7.6,color='#555555')
box(ax,21.6,41.8,14.8,12.4,'커널 측정기','후보 9종 · cold/warm\nCUPTI 시간·그리드\n6,801개 측정 셀','new',ts=8.2,ss=6.5)
for i in range(3):arrow(ax,(18.2,51.4-i*7.2),(21.2,48.0),color=GRAY,lw=.9)
# simulation layer
ax.text(56.0,56.0,'시뮬레이션 계층',ha='center',fontsize=7.6,color='#555555')
box(ax,41.0,45.4,30.0,9.0,'CTA 이벤트 시뮬레이터','요청별 길이 → CTA 실행 시간 예측\nC 구현 · 후보 9개 첫 선택 14.3 ms','new',ts=8.4,ss=6.4)
box(ax,41.0,36.2,14.2,6.8,'Accel-Sim','+ RTX 4090(sm_89)\n자원 민감도','ext',ts=7.8,ss=6.2,pad=.3)
box(ax,56.8,36.2,14.2,6.8,'GPGPU-Sim','+ split-KV 커널\n불균형 기전 재현','ext',ts=7.8,ss=6.2,pad=.3)
arrow(ax,(36.8,49.9),(40.6,49.9),text='보정',toff=(0,1.3))
# outputs
ax.text(86.0,56.0,'산출물',ha='center',fontsize=7.6,color='#555555')
box(ax,76.0,49.4,20.4,5.0,'취약 조건 지도','트레이스 재생 → 빈도·경계','warn',ts=7.8,ss=6.3,pad=.3)
box(ax,76.0,42.2,20.4,5.0,'실측 테이블','측정한 최선 분할 수','soft',ts=7.8,ss=6.3,pad=.3)
box(ax,76.0,35.0,20.4,5.0,'보정된 예측 모델','요청별 길이 → 후보별 시간','soft',ts=7.8,ss=6.3,pad=.3)
ax.plot([34.0,34.0,98.2,98.2],[54.6,58.4,58.4,44.7],color=NAVY,lw=.9,zorder=4);arrow(ax,(98.2,44.7),(97.0,44.7),lw=.9)
ax.text(75.0,58.4,'후보별 실측 시간',ha='center',va='center',fontsize=6.6,color=NAVY,zorder=6,bbox=dict(fc='#F5F8F9',ec='none',pad=.4))
arrow(ax,(71.4,51.9),(75.6,51.9));arrow(ax,(71.4,46.4),(75.6,37.6))
# online selection
y,h=16.4,7.6
box(ax,2.4,y,12.6,h,'연속 배치 엔진','페이지 KV\n최대 배치 64','old',ts=7.6,ss=6.3,pad=.3)
box(ax,18.4,y,12.2,h,'요청별 KV 길이','[32K, 512, …]','old',ts=7.6,ss=6.6,pad=.3)
box(ax,34.0,y,12.4,h,'결정 캐시','256토큰 페이지\n수 벡터가 키','new',ts=7.8,ss=6.3,pad=.3)
box(ax,51.0,y-1.2,21.6,h+2.4,'분할 수 선택','테이블 이웃 + 시뮬레이터 9후보\n혼합 규칙(δ=0.2) → num_splits','new',ts=8.2,ss=6.4)
box(ax,76.6,y,14.4,h,'FlashAttention-2','커널 변경 없음\nnum_splits만 전달','old',ts=7.5,ss=6.2,pad=.3)
ax.text(96.6,y+h/2,'다음\n토큰',ha='center',va='center',fontsize=7.4,fontweight='bold',color=NAVY)
arrow(ax,(15.4,y+h/2),(18.0,y+h/2));arrow(ax,(31.0,y+h/2),(33.6,y+h/2))
arrow(ax,(46.8,y+h/2),(50.6,y+h/2),text='미스',toff=(0,1.2))
arrow(ax,(73.0,y+h/2),(76.2,y+h/2));arrow(ax,(91.4,y+h/2),(94.4,y+h/2))
ax.plot([40.2,40.2,83.8],[y+h+.4,y+h+2.9,y+h+2.9],color=TEAL,lw=1.1,zorder=4);arrow(ax,(83.8,y+h+2.9),(83.8,y+h+.5),color=TEAL,lw=1.1)
ax.text(62,y+h+2.9,'적중: 이전 선택 재사용 (0.9~5.6 µs)',ha='center',va='center',fontsize=6.8,color=TEAL,bbox=dict(fc='#F5F8F9',ec='none',pad=.3),zorder=5)
arrow(ax,(84.0,34.4),(66.0,y+h+1.4),color=NAVY,ls=(0,(3,2)),lw=.9)
ax.text(78.5,27.6,'테이블·모델 적재',ha='left',va='center',fontsize=6.6,color=NAVY)
# verification
for i,t in enumerate(['TPOT · 생성 토큰 일치','분기 분류\n(teacher forcing·커널 탐침)','verify 1,194개\n(GPU 없이 재계산)','대시보드 · 시연']):
 box(ax,13.0+i*21.0,2.0,18.0,5.4,t,'','soft',ts=7.0,pad=.3)
arrow(ax,(96.6,y-.6),(96.6,9.0),color=GRAY,lw=.9)
# legend below the bands
for x0,(st,lab) in zip((10,28,47,72),[('new','새로 구현'),('ext','기존 도구 확장'),('old','기존 도구·엔진 그대로'),('warn','문제(취약 조건)')]):
 s_=STYLE[st]
 ax.add_patch(Rectangle((x0,-3.4),2.2,1.6,fc=s_['fc'],ec=s_['ec'],lw=s_['lw']*.7,zorder=3))
 ax.text(x0+3.0,-2.6,lab,fontsize=7,va='center')
save(fig,'framework')

# ---------------------------------------------------------------- Figure: busy SMs over time, split 1 vs 16
grid=np.linspace(0,3600,721)
fig,ax=plt.subplots(figsize=(7.7,2.45))
for v,lab,col in [('fa2','분할 1 (기본 휴리스틱)',NAVY),('fd_s16','분할 16',TEAL)]:
 a=active_sms(v,grid);ax.fill_between(grid,a,step='post',color=col,alpha=.25 if v=='fa2' else .55,lw=0)
 ax.step(grid,a,where='post',color=col,lw=1.4,label=lab)
ax.annotate('짧은 요청이 끝난 뒤 긴 요청의 CTA 8개\n(SM 4개)만 일하는 꼬리 구간',xy=(1800,8),xytext=(1500,60),fontsize=8.3,color=NAVY,
 arrowprops=dict(arrowstyle='->',color=NAVY,lw=1))
ax.annotate('모든 SM이 함께 처리하고 일찍 종료',xy=(200,110),xytext=(520,105),fontsize=8.3,color=TEAL,
 arrowprops=dict(arrowstyle='->',color=TEAL,lw=1))
ax.set_xlim(0,3600);ax.set_ylim(0,135);ax.set_yticks([0,32,64,96,128])
ax.set_xlabel('커널 실행 시간 (µs)',fontsize=9.5);ax.set_ylabel('일하는 SM 수',fontsize=9.5)
ax.legend(frameon=False,fontsize=9,loc='upper right');ax.grid(alpha=.2)
fig.tight_layout();save(fig,'sm_timeline')

# Figure: keys per CTA for 32K x1 + 512 x31 (one KV head), split 1 vs 16.
fig,axs=plt.subplots(1,2,figsize=(7.7,2.35),sharey=True)
lens=[32768]+[512]*31
for ax,s,col in zip(axs,[1,16],[C[0],C[1]]):
 width=32768/s
 work=[max(0,min(n-k*width,width)) for n in lens for k in range(s)]
 ax.bar(np.arange(len(work)),np.array(work)/1024,color=col,width=1.0 if s>1 else .8)
 ax.set_title(f'분할 수 {s}  (CTA {len(work)}개)',fontsize=11)
 ax.set_xlabel('요청 순서로 나열한 CTA',fontsize=9)
 ax.set_xticks([]);ax.set_ylim(0,35);ax.grid(axis='y',alpha=.2)
 share=sum(lens)*8/128/1024
 ax.axhline(share,color=C[2],ls='--',lw=1.1)
 ax.text(len(work)*.98,share+1.2,'SM 1개의 균등 몫',ha='right',fontsize=8.5,color=C[2])
axs[0].set_ylabel('CTA당 키 수 (천 개)')
fig.tight_layout();save(fig,'imbalance_v2')

# Figure: cost of one evaluation per tool (seconds, log axis); values from the stored records.
rows=[('Accel-Sim: FA2 B=16 커널 재생',992.58,C[2]),('GPGPU-Sim: split-KV 축소 커널 실행',330,'#D9A066'),('Accel-Sim: FD B=1·L=8K 커널 재생',200.34,C[2]),
 ('Accel-Sim: FD B=1·L=1K 커널 재생',19.06,C[2]),('CTA 시뮬레이터(NumPy) 첫 선택',0.8426,C[3]),
 ('CTA 시뮬레이터(C) 첫 선택',0.014293,C[0]),('결정 캐시 적중',5.567e-6,C[1])]
fig,ax=plt.subplots(figsize=(7.7,3.0))
y=np.arange(len(rows))[::-1]
ax.barh(y,[r[1] for r in rows],color=[r[2] for r in rows],height=.62)
ax.set_xscale('log');ax.set_yticks(y,labels=[r[0] for r in rows],fontsize=9.5)
for yy,(_,v,_) in zip(y,rows):
 txt=(f'{v:,.0f} s' if v==int(v) else f'{v:,.1f} s') if v>=1 else (f'{v*1e3:.1f} ms' if v>=1e-3 else f'{v*1e6:.1f} µs')
 ax.text(v*1.35,yy,txt,va='center',fontsize=9,bbox=dict(fc='white',ec='none',pad=.5))
ax.axvline(0.03425,color='gray',ls='--',lw=1,zorder=0);ax.text(0.03425*1.25,len(rows)-.55,'디코드 1단계(34 ms)',fontsize=8.5,color='#555555')
ax.set_xlim(1e-6,2e4)
ax.set_xticks([1e-6,1e-4,1e-2,1,1e2,1e4],labels=['1 µs','100 µs','10 ms','1 s','100 s','10,000 s']);ax.set_xlabel('1회 평가에 걸린 시간 (초, 로그 축)');ax.grid(axis='x',alpha=.2)
fig.tight_layout();save(fig,'sim_cost')
print('wrote concept, method_map, framework, sm_timeline, imbalance_v2, sim_cost')
