"""Original dark chapter banners; conceptual pictures, not measurements."""
from pathlib import Path
from html import escape
import json,re,sys
sys.path.insert(0,str(Path(__file__).parent/'illustrations'))
from drawing import ROOT,FONT_STYLE,LICENSE
TEAL='#2dd4bf';GOLD='#fbbf24';GREEN='#a3e635';WHITE='#edf8f6';MUTED='#adcbce'
SPECS=[
('Types meet','machine code.','Specialization at the JIT boundary','flow',['Python','JIT','Kernel']),
('Choose the tile.','Keep the contract.','Problem shape versus launch configuration','flow',['N','BLOCK','Grid']),
('Same tensor.','Different addresses.','Strides connect coordinates to storage','grid',['(i, j)','Strides','Offset']),
('One grid.','Many programs.','Programs cooperate on logical tiles','grid',['Grid','Programs','Tiles']),
('Work in blocks.','Guard the tail.','Vectorized lanes with bounded accesses','tail',['Valid','Valid','Masked']),
('Pointers carry','responsibility.','Device, dtype, stride and bounds','flow',['Tensor','Contract','Pointer']),
('Expand the view.','Preserve ownership.','Broadcasting and block-pointer access','grid',['Rows','Columns','Tile']),
('Allocate outside.','Compute inside.','Host-managed storage for GPU programs','flow',['Allocate','Launch','Fill']),
('Structure is','a predicate.','Causal, diagonal and sparse masks','causal',['Queries','Mask','Keys']),
('Logical tiles.','Physical hardware.','Compiler-managed mapping to threads','flow',['Tile','Layout','Threads']),
('Budget the live','working set.','Registers, shared memory and reuse','resources',['Registers','Shared','Cache']),
('Fuse the work.','Skip the traffic.','Elementwise operations in one kernel','flow',['Load','f(x)','Store']),
('Tile the product.','Accumulate K.','Blocked matrix multiplication','gemm',['A','B','C']),
('Many lanes.','One answer.','Reduction identities and valid counts','reduce',['Values','Reduce','Scalar']),
('A kernel joins','the graph.','Custom autograd with a clear boundary','flow',['Forward','Save','Backward']),
('Follow the','chain rule.','Exact derivatives and upstream gradients','flow',['dy','Local dx','Gradient']),
('Paths branch.','Gradients add.','Fan-out forward, accumulation backward','branch',['x','Two paths','Sum dx']),
('Try candidates.','Keep the winner.','Correctness before GPU timing','flow',['Configs','Measure','Select']),
('Wait for work.','Then trust time.','Device-aware timing and warm-up','timeline',['Warm up','Measure','Report']),
('Normalize rows.','Fuse the layer.','Reductions, affine work and data reuse','reduce',['Row','Mean/var','Output']),
('Stream scores.','Keep the state.','Online attention and top-2 MoE gating','attention',['Q / K','m, l, acc','O']),
('Model payoffs.','Check uncertainty.','Pricing, rolling statistics and simulation','plot',['Paths','Payoffs','Mean']),
]
class Banner:
 def __init__(self):self.parts=[]
 def text(self,x,y,value,size=18,color=WHITE,anchor='start',weight=400):self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')
 def box(self,x,y,w,h,label='',color=TEAL):
  self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="#103238" stroke="{color}" stroke-width="1.6"/>')
  if label:self.text(x+w/2,y+h/2+6,label,17,color,'middle',600)
 def arrow(self,x,y,xx,yy,color=TEAL):self.parts.append(f'<path d="M{x} {y} L{xx} {yy}" stroke="{color}" stroke-width="3" fill="none" marker-end="url(#arrow)"/>')
 def grid(self,x,y,rows=4,cols=6,cell=25,causal=False,tail=False):
  for i in range(rows):
   for j in range(cols):
    on=(not causal or j<=i) and (not tail or i==0 or j<3)
    self.parts.append(f'<rect x="{x+j*cell}" y="{y+i*cell}" width="{cell-4}" height="{cell-4}" rx="3" fill="{TEAL if on else "#375156"}" opacity="{.9 if on else .5}"/>')
 def diagram(self,kind,labels):
  if kind in ['flow','timeline']:
   for i,label in enumerate(labels):
    x=370+i*116;self.box(x,133,98,78,label,GOLD if i==1 else TEAL)
    if i<2:self.arrow(x+99,172,x+113,172)
   if kind=='timeline':
    self.arrow(373,250,696,250,GOLD);self.text(535,280,'Observe completion',18,MUTED,'middle')
   else:self.text(535,263,'A boundary you can reason about',17,MUTED,'middle')
  elif kind in ['grid','tail','causal']:
   self.box(370,102,112,140,labels[0],GOLD);self.arrow(486,173,530,173)
   self.grid(548,112,4,6,25,kind=='causal',kind=='tail');self.text(550,276,labels[1]+' → '+labels[2],18,MUTED,'middle')
  elif kind=='gemm':
   self.grid(370,105,4,3,20);self.text(400,220,'A tiles',18,TEAL,'middle');self.text(447,155,'×',28,GOLD,'middle')
   self.grid(470,105,3,4,20);self.text(507,220,'B tiles',18,TEAL,'middle');self.arrow(555,151,594,151)
   self.grid(609,105,4,4,22);self.text(651,220,'C tile',18,GOLD,'middle');self.text(537,275,'Reuse inputs along K',18,MUTED,'middle')
  elif kind=='resources':
   for i,label in enumerate(labels):self.box(385+i*105,105+i*33,125,63,label,[TEAL,GOLD,GREEN][i])
   self.text(542,285,'Live values determine cost',18,MUTED,'middle')
  elif kind=='reduce':
   self.grid(370,130,1,6,23);self.arrow(517,141,554,174);self.box(569,148,130,70,labels[1],GOLD);self.text(535,270,labels[0]+' → '+labels[2],18,MUTED,'middle')
  elif kind=='branch':
   self.box(365,145,65,65,'x',GOLD);self.box(486,93,94,61,'x × x');self.box(486,210,94,61,'+ x');self.box(634,145,75,65,'dx',GREEN)
   for a,b,c,d in [(431,175,480,125),(431,179,480,239),(584,125,628,173),(584,239,628,179)]:self.arrow(a,b,c,d)
   self.text(538,310,'Accumulate both paths',18,MUTED,'middle')
  elif kind=='attention':
   self.grid(366,109,4,4,23,True);self.arrow(463,155,505,155);self.box(517,116,102,80,'m, l, acc',GOLD);self.arrow(624,155,653,155);self.box(665,124,50,64,'O',GREEN);self.text(540,270,'Rescale both running sums',18,MUTED,'middle')
  elif kind=='plot':
   self.parts.append('<path d="M380 235 H700 M380 235 V95" stroke="#6e969c" stroke-width="2" fill="none"/>')
   for i,ys in enumerate([[218,198,208,158,170,115],[220,215,180,191,142,155],[217,192,168,175,117,101]]):
    points=' '.join(f'{390+j*55},{y}' for j,y in enumerate(ys));self.parts.append(f'<polyline points="{points}" fill="none" stroke="{[TEAL,GOLD,GREEN][i]}" stroke-width="2.5"/>')
   self.text(540,280,'Illustrative paths, not market data',17,MUTED,'middle')
manifest=[]
for n,(a,b,subtitle,kind,labels) in enumerate(SPECS,1):
 d=Banner();d.parts.append('<rect x="1" y="1" width="758" height="338" rx="19" fill="url(#bg)" stroke="#55747c"/><rect x="1" y="1" width="758" height="338" rx="19" fill="url(#dots)"/><path d="M29 44 V280" stroke="#a3e635" stroke-width="3"/>')
 d.text(43,65,f'CHAPTER {n:02}',13,GREEN,weight=700);d.text(43,118,a,27,weight=700);d.text(43,154,b,27,weight=700)
 # Keep subtitles short and measured, with explicit words-per-line wrapping.
 from PIL import ImageFont
 font=ImageFont.truetype(str(ROOT/'tools/assets/Inter.ttf'),18)
 lines=[];line=''
 for word in subtitle.split():
  trial=(line+' '+word).strip()
  if font.getlength(trial)>282:lines.append(line);line=word
  else:line=trial
 if line:lines.append(line)
 for i,line in enumerate(lines):d.text(43,207+i*25,line,18,MUTED)
 d.text(43,307,'TRITON FROM FIRST PRINCIPLES',11,MUTED);d.diagram(kind,labels)
 title=f'Chapter {n}: {a} {b}'
 svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="340" viewBox="0 0 760 340" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}. Conceptual chapter illustration.</desc>{LICENSE}{FONT_STYLE}<defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#07191c"/><stop offset="1" stop-color="#12383a"/></linearGradient><pattern id="dots" width="19" height="19" patternUnits="userSpaceOnUse"><circle cx="5" cy="5" r=".7" fill="#91b7b7" opacity=".14"/></pattern><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0 L10 5 L0 10" fill="none" stroke="{TEAL}" stroke-width="2"/></marker></defs><g font-family="BookDiagram, Inter, Arial, sans-serif">{''.join(d.parts)}</g></svg>'''
 rel=f'images/chapter-openers/{n:02}.svg';p=ROOT/'docs'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(svg);manifest.append(dict(chapter=n,path='docs/'+rel,title=title))
 for page in (ROOT/'docs').glob('part*/*.md'):
  s=page.read_text()
  if re.match(rf'# Chapter {n}:',s):
   marker=f'\n[![{title}](../{rel})](../{rel})\n'
   if marker not in s:
    idx=s.index('\n');page.write_text(s[:idx+1]+marker+s[idx+1:])
(ROOT/'tools/illustrations/chapter-openers.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Created and linked 22 dark chapter banners.')

# Entry pages use the same visual system, with their own learning-path diagrams.
for page,name,kicker,lines,subtitle,labels in [
 ('index.md','book-overview','BOOK OVERVIEW',['From tensors.','To GPU kernels.'],['Understand the contract.','Build, differentiate and validate.'],['Contract','Kernel','Evidence']),
 ('getting-started.md','getting-started','START HERE',['Build the stack.','Verify each layer.'],['Driver, Python environment','and the first checked launch.'],['Driver','Runtime','GPU test']),
]:
 d=Banner();d.parts.append('<rect x="1" y="1" width="758" height="338" rx="19" fill="url(#bg)" stroke="#55747c"/><rect x="1" y="1" width="758" height="338" rx="19" fill="url(#dots)"/><path d="M29 44 V280" stroke="#a3e635" stroke-width="3"/>')
 d.text(43,65,kicker,13,GREEN,weight=700)
 for i,line in enumerate(lines):d.text(43,118+i*36,line,27,weight=700)
 for i,line in enumerate(subtitle):d.text(43,207+i*25,line,18,MUTED)
 d.text(43,307,'TRITON FROM FIRST PRINCIPLES',11,MUTED);d.diagram('flow',labels)
 title=' '.join(lines);description=' '.join(subtitle)
 # Reuse only the font/license and background definitions, never chapter content.
 template=(ROOT/'docs/images/chapter-openers/01.svg').read_text()
 template=re.sub(r'<title id="title">.*?</title>',f'<title id="title">{escape(title)}</title>',template)
 template=re.sub(r'<desc id="desc">.*?</desc>',f'<desc id="desc">{escape(description)} Conceptual learning path.</desc>',template)
 template=re.sub(r'<g font-family="BookDiagram, Inter, Arial, sans-serif">.*?</g>',lambda m:'<g font-family="BookDiagram, Inter, Arial, sans-serif">'+''.join(d.parts)+'</g>',template,flags=re.S)
 rel=f'images/chapter-openers/{name}.svg';(ROOT/'docs'/rel).write_text(template)
 p=ROOT/'docs'/page;s=p.read_text();marker=f'\n[![{title}]({rel})]({rel})\n'
 if marker not in s:
  at=s.index('\n');p.write_text(s[:at+1]+marker+s[at+1:])
print('Created and linked Home and Getting Started overview banners.')
