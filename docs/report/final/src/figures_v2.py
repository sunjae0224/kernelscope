"""Figures added for the restructured final report (v2); the shared ones come from figures.py."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont('/root/.local/share/fonts/malgun.ttf')
font_manager.fontManager.addfont('/root/.local/share/fonts/malgunbd.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
C=['#284B63','#4C8F95','#BD7339','#7B6E92']
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),facecolor='white',bbox_inches='tight',pad_inches=.12)
 fig.savefig(OUT/(name+'.svg'),facecolor='white',bbox_inches='tight',pad_inches=.12)
 plt.close(fig)

# Figure 1: six stages, snake layout.
fig,ax=plt.subplots(figsize=(7.7,2.6));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
labels=[('커널 실측','RTX 4090 후보별 시간·CTA 구성'),('시뮬레이션 분석','성능·사이클 시뮬레이터'),
 ('취약 조건 관측','혼합 길이·배치 문턱·트레이스'),('분할 수 동적 선택','테이블·성능 모델·혼합 정책'),
 ('실제 모델 실행','Qwen3-4B·Llama 8B의 TPOT'),('출력·재현 검증','토큰 분기·로짓·수치 재계산')]
positions=[(.1,1.8),(3.5,1.8),(6.9,1.8),(6.9,.15),(3.5,.15),(.1,.15)]
for i,((h,t),(x,y)) in enumerate(zip(labels,positions)):
 fc='#DCE8EC' if i==1 else '#EDF2F4'
 ax.add_patch(FancyBboxPatch((x,y),2.95,.9,boxstyle='round,pad=0.07',fc=fc,ec=C[0],lw=1.1))
 ax.text(x+1.475,y+.59,h,ha='center',va='center',fontsize=12,fontweight='bold')
 ax.text(x+1.475,y+.24,t,ha='center',va='center',fontsize=8.6)
for a,b in [(0,1),(1,2),(3,4),(4,5)]:
 x,y=positions[a];xx,yy=positions[b]
 ax.annotate('',xy=(xx-.1 if xx>x else xx+3.05,yy+.45),xytext=(x+3.05 if xx>x else x-.1,y+.45),arrowprops=dict(arrowstyle='->',color=C[0],lw=1.4))
ax.annotate('',xy=(8.375,1.12),xytext=(8.375,1.7),arrowprops=dict(arrowstyle='->',color=C[0],lw=1.4))
save(fig,'pipeline_v2')

# Figure 2: keys per CTA for 32K x1 + 512 x31 (one KV head), split 1 vs 16.
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

# Figure 3: cost of one evaluation per tool (seconds, log axis); values from the stored records.
rows=[('Accel-Sim: FA2 B=16 커널 재생',992.58,C[2]),('GPGPU-Sim: split-KV 축소 커널 실행',330,'#D9A066'),('Accel-Sim: FD B=1·L=8K 커널 재생',200.34,C[2]),
 ('Accel-Sim: FD B=1·L=1K 커널 재생',19.06,C[2]),('성능 시뮬레이터(NumPy) 첫 선택',0.8426,C[3]),
 ('성능 시뮬레이터(C) 첫 선택',0.014293,C[0]),('결정 캐시 적중',5.567e-6,C[1])]
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
print('wrote pipeline_v2, imbalance_v2, sim_cost')
