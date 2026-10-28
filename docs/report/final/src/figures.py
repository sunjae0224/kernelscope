"""Original report figures; numeric panels read committed experiment records."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parent.parent
REPO=ROOT.parents[2]
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont('/root/.local/share/fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
C=['#284B63','#4C8F95','#BD7339','#7B6E92']
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),facecolor='white',bbox_inches='tight',pad_inches=.12)
 fig.savefig(OUT/(name+'.svg'),facecolor='white',bbox_inches='tight',pad_inches=.12)
 plt.close(fig)

fig,ax=plt.subplots(figsize=(7.7,2.6));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
labels=[('프로파일링','요청별 길이·커널 실측'),('시뮬레이터 개선','CTA 이벤트·C 계산 경로'),('엣지 조건 관측','부하 불균형·캐시·부하'),('동적 설정 선택','KV 분할·변형·선택 비용'),('실제 모델 검증','TPOT·토큰·로짓 비교'),('근거 재계산','기록·재현 검증·대시보드')]
positions=[(.1,1.8),(3.5,1.8),(6.9,1.8),(6.9,.15),(3.5,.15),(.1,.15)]
for (h,t),(x,y) in zip(labels,positions):
 ax.add_patch(FancyBboxPatch((x,y),2.95,.9,boxstyle='round,pad=0.07',fc='#EDF2F4',ec=C[0],lw=1.1))
 ax.text(x+1.475,y+.59,h,ha='center',va='center',fontsize=12,fontweight='bold')
 ax.text(x+1.475,y+.24,t,ha='center',va='center',fontsize=9)
for a,b in [(0,1),(1,2),(3,4),(4,5)]:
 x,y=positions[a];xx,yy=positions[b]
 ax.annotate('',xy=(xx-.1 if xx>x else xx+3.05,yy+.45),xytext=(x+3.05 if xx>x else x-.1,y+.45),arrowprops=dict(arrowstyle='->',color=C[0],lw=1.4))
ax.annotate('',xy=(8.375,1.12),xytext=(8.375,1.7),arrowprops=dict(arrowstyle='->',color=C[0],lw=1.4))
save(fig,'pipeline')

fig,axs=plt.subplots(1,2,figsize=(7.7,2.35),sharey=True)
for ax,s in zip(axs,[1,8]):
 lens=[32768]+[512]*7
 work=[]
 for length in lens:
  width=32768/s
  work.extend([max(0,min(length-k*width,width)) for k in range(s)])
 ax.bar(np.arange(len(work)),np.array(work)/1024,color=C[0] if s==1 else C[1],width=.9)
 ax.set_title(f'분할 {s}: 요청별 유효 작업량',fontsize=11)
 ax.set_xlabel('요청 순서로 나열한 CTA 작업 조각',fontsize=9)
 ax.set_xticks([]);ax.set_ylim(0,35);ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('유효 KV 길이 (K 토큰)')
fig.tight_layout();save(fig,'imbalance')

with open(REPO/'docs/experiments/gpgpusim-splitkv-sweep.csv') as f:rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(7.7,2.6))
for i,(key,label) in enumerate([('ragged','혼합 2048+128×15'),('uniform','균일 256×16')]):
 rr=[r for r in rows if r['workload']==key];rr.sort(key=lambda r:int(r['S']))
 ax.plot([int(r['S']) for r in rr],[float(r['gpu_sim_cycle'])/1000 for r in rr],marker='o',label=label,color=C[i],lw=2)
ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],labels=['1','2','4','8','16'])
ax.set_xlabel('KV 분할 수');ax.set_ylabel('실행 사이클 (천)');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=10)
fig.tight_layout();save(fig,'sim_splits')

j=json.loads((ROOT/'evidence/results.json').read_text())
def lookup(g,p):return next(r for r in g['rows'] if r['policy']==p)
groups=[j['fixed_split_serving'][x] for x in ['uniform','ragged','arrivals']]+[j['long_context_text']]
x=np.arange(4);fig,ax=plt.subplots(figsize=(7.7,2.6))
for off,pol,label,col in [(-.19,'heuristic','기본 휴리스틱',C[0]),(.19,'table','테이블 선택',C[1])]:
 vals=[lookup(g,pol)['tpot_ms_mean'] for g in groups]
 bars=ax.bar(x+off,vals,.36,label=label,color=col)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',va='bottom',fontsize=9)
ax.set_xticks(x,labels=['균일','32K 혼합','요청 도착','64K 반복 자연어']);ax.set_ylabel('TPOT (ms)');ax.set_ylim(0,115)
ax.legend(frameon=False,ncol=2,fontsize=10);ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'tpot')

rows=j['traffic_sweep']['rows']
fig,ax=plt.subplots(figsize=(7.7,2.65))
for i,(key,label) in enumerate([('azure_conv','Azure 대화'),('azure_code','Azure 코드'),('burstgpt','BurstGPT')]):
 rr=[r for r in rows if key.lower() in str(r.get('trace','')).lower()]
 if not rr:
  # The audit preserves original names (conv / code / BurstGPT).
  needle={'azure_conv':'conv','azure_code':'code','burstgpt':'burst'}[key]
  rr=[r for r in rows if needle in str(r.get('trace','')).lower()]
 rr.sort(key=lambda r:float(r['batch_mean']))
 if rr:ax.plot([float(r['batch_mean']) for r in rr],[100*float(r['loss_ge_1.25_frac']) for r in rr],marker='o',ms=4,lw=1.5,label=label,color=C[i])
ax.axvline(26,color='gray',ls='--',lw=1);ax.text(26.5,14.8,'해당 구성의 분할 없음 문턱',fontsize=9,color='#555555')
ax.set_xlabel('평균 활성 배치 크기');ax.set_ylabel('시간 비 ≥1.25인 단계 (%)');ax.set_ylim(-.3,16.5);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=10,loc='center right')
fig.tight_layout();save(fig,'load_sweep')
print('wrote 5 original PNG/SVG figure pairs')
