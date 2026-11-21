"""Original appendix banners and concept diagrams; no external artwork or GPU runs."""
import runpy,re,sys,json
from pathlib import Path
from html import escape
NS=runpy.run_path(str(Path(__file__).with_name('create_chapter_openers.py')))
Banner,ROOT=NS['Banner'],NS['ROOT']
sys.path.insert(0,str(Path(__file__).parent/'illustrations'))
from drawing import Drawing
SPECS=[
 ('A','installation-setup.md','Check the layer.','Find the failure.','Driver, runtime and a checked launch.',['Driver','Runtime','Probe']),
 ('B','practice-quiz.md','Trace by hand.','Then test the claim.','Reason through shapes, masks and arithmetic.',['Trace','Predict','Check']),
 ('C','pytorch-triton-integration.md','Connect the kernel.','Keep its contract.','Execution, gradients and graph capture.',['Tensor','Kernel','Result']),
 ('D','memory-model-continued.md','Describe the tile.','Respect the layout.','Grouped hints, pointers and descriptors.',['Offsets','Layout','Access']),
]
manifest=[]
for letter,page,a,b,description,labels in SPECS:
 d=Banner();d.parts.append('<rect x="1" y="1" width="758" height="338" rx="19" fill="url(#bg)" stroke="#55747c"/><rect x="1" y="1" width="758" height="338" rx="19" fill="url(#dots)"/><path d="M29 44 V280" stroke="#a3e635" stroke-width="3"/>')
 d.text(43,65,'APPENDIX '+letter,13,NS['GREEN'],weight=700);d.text(43,118,a,27,weight=700);d.text(43,154,b,27,weight=700)
 from drawing import wrap
 for i,line in enumerate(wrap(description,278,18)):d.text(43,207+i*25,line,18,NS['MUTED'])
 d.text(43,307,'TRITON FROM FIRST PRINCIPLES',11,NS['MUTED']);d.diagram('flow',labels)
 template=(ROOT/'docs/images/chapter-openers/01.svg').read_text();title=a+' '+b
 template=re.sub(r'<title id="title">.*?</title>',f'<title id="title">Appendix {letter}: {escape(title)}</title>',template)
 template=re.sub(r'<desc id="desc">.*?</desc>',f'<desc id="desc">{escape(description)} Conceptual appendix illustration.</desc>',template)
 template=re.sub(r'<g font-family="BookDiagram, Inter, Arial, sans-serif">.*?</g>',lambda m:'<g font-family="BookDiagram, Inter, Arial, sans-serif">'+''.join(d.parts)+'</g>',template,flags=re.S)
 banner=f'images/chapter-openers/appendix-{letter.lower()}.svg';(ROOT/'docs'/banner).write_text(template)
 rel=f'appendix-{letter.lower()}/01-concept.svg';(ROOT/'docs/images'/f'appendix-{letter.lower()}').mkdir(exist_ok=True)
 if letter=='A':
  w=Drawing(rel,'Isolate the first failing layer','A successful earlier layer is evidence only for that layer.')
  w.flow([('Driver and device','nvidia-smi checks visibility and the driver. Preserve a working provider driver.','blue'),('Runtime environment','Check CPU architecture and Python/PyTorch/Triton versions. CUDA availability does not prove kernel compilation.','violet'),('Checked execution','Unset interpreter mode for a GPU test. Compare complete output against an oracle and preserve commands, dates and logs.','teal')])
 elif letter=='B':
  w=Drawing(rel,'A worked answer needs a contract and a trace','Example: N=13, BLOCK=8. Decide which offsets may touch memory.')
  w.matrix([list(range(8)),list(range(8,13))+['off']*3],cell=82,colors=[['teal']*8,['blue']*5+['gray']*3])
  w.note('Check coverage and safety','The valid offsets are 0 through 12 exactly once. Offsets 13,14,15 must be masked on memory accesses. A correct shape alone does not prove these bounds.')
 elif letter=='C':
  w=Drawing(rel,'Three integration questions, three contracts','A launch, an autograd derivative and graph capture are separate capabilities.')
  w.flow([('Execute','The host allocates storage and launches a kernel with checked device, dtype, shape and strides.','blue'),('Differentiate','Provide a correct backward through an autograd wrapper or an appropriate registered formula. A raw launch does not supply derivatives.','violet'),('Capture and compose','Supported direct launches can be captured by torch.compile. triton_op adds structured subsystem integration; it is not mandatory for every direct launch.','teal')])
 else:
  w=Drawing(rel,'Hints describe every group, not just the first','Contiguity four: each colored run contains four sequential offsets.')
  w.matrix([[0,1,2,3,8,9,10,11]],cell=82,colors=[['blue']*4+['teal']*4])
  w.note('Trace the starts','The two group starts are 0 and 8. Both are divisible by four; the intermediate values 1,2,3 are not. This distinguishes group-start divisibility from an every-element claim.')
  w.columns([('Block pointer','Describe the logical tile, strides, offsets and boundary checks.','blue'),('Descriptor','Also meet base/stride alignment, contiguous final dimension, supported shape and allocator requirements.','violet')])
 height=w.write();path=ROOT/'docs/images'/rel;s=path.read_text().replace('CHAPTER '+letter,'APPENDIX '+letter);# Drawing path metadata is made explicit for appendices.
 s=re.sub(r'CHAPTER pendix-[a-d]', 'APPENDIX '+letter,s);path.write_text(s)
 p=ROOT/'docs/appendix'/page;s=p.read_text();links=f'\n[![Appendix {letter}: {title}](../{banner})](../{banner})\n\n[![Appendix {letter} worked concept](../images/{rel})](../images/{rel})\n'
 if f'../{banner}' not in s:
  at=s.index('\n');p.write_text(s[:at+1]+links+s[at+1:])
 manifest.append(dict(appendix=letter,banner='docs/'+banner,concept='docs/images/'+rel,height=height))
(ROOT/'tools/illustrations/appendices.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Added four appendix banners and four explanatory SVGs.')
