"""Create original, numeric chapter diagrams. No GPU or external artwork."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).parent/'illustrations'))
from drawing import Drawing,ROOT

SOURCE='Triton language documentation; illustrative worked arithmetic, not a GPU measurement.'
SPECS=[
(1,'Compile a signature, then launch its programs','Python prepares the launch; the compiler specializes the kernel.', [('Python host','Pointers, scalar arguments and launch metadata.','blue'),('Compilation key','Types, constexpr values, target and compiler specialization.','violet'),('GPU launch','A cache hit reuses an artifact; it does not skip execution.','teal')]),
(2,'Separate the problem from its tile configuration','Changing a constexpr can change generated code; changing a tensor shape alone need not.', [('Problem','Logical length N and valid output elements.','blue'),('Configuration','BLOCK=128 or 256; num_warps and num_stages.','violet'),('Launch grid','ceil(N / BLOCK) programs, with a final bounds mask.','teal')]),
(3,'A logical coordinate is not a byte address','A padded row-major tensor has shape (2,3), element strides (5,1).',None),
(4,'A launch grid is a grid of program instances','Each program computes a logical tile; physical threads cooperate on that tile.',None),
(5,'Vectorize the block, mask the tail','N=13 values, BLOCK=8: two programs cover 16 logical lanes.',None),
(6,'The caller owns the pointer-and-stride contract','A pointer does not carry a checked tensor shape into the kernel.', [('Host tensor','Check device, dtype, shape, strides and storage bounds.','blue'),('Kernel arguments','Pass the pointer and every stride the indexing formula needs.','violet'),('Address calculation','offset(i,j) = i*stride_i + j*stride_j; mask invalid coordinates.','teal')]),
(7,'Broadcasting expands an expression, not its storage','A column (2,1) plus a row (1,3) creates a (2,3) logical result.',None),
(8,'Allocate on the host, populate on the device','The wrapper owns storage; the kernel owns the computation over it.', [('PyTorch allocator','Create input and output tensors on the intended CUDA device.','blue'),('Triton kernel','Construct logical offsets; load, calculate and store.','teal'),('Stream dependency','Consume the output after its producing work, on an ordered stream.','violet')]),
(9,'A causal mask includes the diagonal','For top-left causal attention, key j is allowed when j <= query i.',None),
(10,'Logical blocks are distributed over physical threads','BLOCK is an array extent, not a count of registers in one thread.', [('One program tile','A block-shaped value is spread according to a compiler layout.','blue'),('Warps and lanes','num_warps selects cooperating warps on NVIDIA; each has 32 lanes.','violet'),('An SM','Registers, shared memory and scheduling limit residency. Measure the result.','teal')]),
(11,'Balance live values against resident work','Larger tiles can improve reuse while increasing resource demand.', [('Registers','Hold distributed live values; the live set matters more than array syntax.','violet'),('Shared memory','Compiler-generated staging and communication can consume on-chip capacity.','teal'),('Evidence','Inspect compiled metadata and profile; the interpreter models neither occupancy nor spills.','blue')]),
(12,'Fuse elementwise work without changing its equation','One launch can remove intermediate global-memory traffic.', [('Input x','Load only valid elements.','blue'),('Compute','Example: y = max(2*x + 1, 0). Keep intermediates local.','violet'),('Output y','Store once; compare every value against an independent reference.','teal')]),
(13,'Every output sums the complete K dimension','For two 2x2 matrices, trace one output before tiling the full problem.',None),
(14,'Padding needs the reduction identity','A masked lane must contribute the neutral value for the selected operation.',None),
(15,'Autograd connects a forward to a backward contract','PyTorch tracks the graph; a custom Triton launch does not invent its derivative.', [('Forward','Compute y, save the inputs needed by the backward formula.','blue'),('Upstream gradient','grad_output carries the derivative of the loss with respect to y.','violet'),('Backward','Return grad_output * dy/dx for each differentiable input.','teal')]),
(16,'Differentiate the function actually implemented','For exact GELU, differentiate x times the standard normal CDF.', [('Forward','y = x*Phi(x); Phi(x) = (1+erf(x/sqrt(2)))/2.','blue'),('Local derivative','dy/dx = Phi(x) + x*phi(x); phi(x) = exp(-x^2/2)/sqrt(2*pi).','violet'),('Chain rule','dx = grad_output * dy/dx; check non-unit upstream gradients too.','teal')]),
(17,'Fan-in adds gradient contributions','At x=2, y=x*x+x=6. The total derivative is 2*x+1=5.',None),
(18,'Autotuning chooses among correct candidates','A configuration must pass correctness checks before timing is useful.', [('Candidate set','Tile extents, warps and pipeline stages.','blue'),('Per-key measurements','Warm up, reset mutated outputs and measure on the actual GPU.','violet'),('Selected configuration','Cache the measured choice for its key; it is not universal across GPUs.','teal')]),
(19,'Time the intended interval, not host enqueue calls','GPU work is asynchronous. A benchmark must observe completion.', [('Prepare','Compile, warm up, choose cache policy and reset state.','blue'),('Measure','Use device events or a GPU-aware benchmark; include all relevant streams.','violet'),('Report','State shapes, dtype, device, versions and statistics. Check the output again.','teal')]),
(20,'Layer normalization reduces within each row','Use the population variance over valid features; padded lanes are not samples.',None),
(21,'Stream attention without storing the whole score matrix','Keep a running maximum, denominator and weighted-value numerator.',None),
(22,'Monte Carlo pricing is a mean of discounted payoffs','Illustrative terminal prices [80,100,120], strike 100; not a simulated experiment.',None),
]

manifest=[]
def make(n,title,lede,items):
 path=f'ch{n:02}/01-concept.svg';(ROOT/'docs/images'/f'ch{n:02}').mkdir(parents=True,exist_ok=True)
 d=Drawing(path,title,lede,SOURCE)
 if items:d.flow(items)
 elif n==3:
  d.matrix([[0,1,2],[5,6,7]],cell=104,rowlabels=['i=0','i=1'],collabels=['j=0','j=1','j=2'],caption='Cells contain element offsets. Two unused storage positions separate the rows.')
  d.note('Trace (1,2)','1*5 + 2*1 = 7 elements. For FP32, the byte displacement is 7*4 = 28 bytes.')
 elif n==4:
  d.matrix([['(0,0)','(0,1)','(0,2)'],['(1,0)','(1,1)','(1,2)']],cell=146,caption='Six programs. program_id(0) and program_id(1) select a tile; they are not CUDA thread IDs.')
 elif n==5:
  d.matrix([list(range(8)),list(range(8,13))+['off']*3],colors=[['teal']*8,['blue']*5+['gray']*3],cell=82,caption='The second program masks offsets 13, 14 and 15 before memory access.')
  d.note('A mask protects an access','Use mask=offsets < N on loads and stores. Computing a masked offset alone does not access memory.')
 elif n==7:
  d.matrix([[11,21,31],[12,22,32]],cell=112,rowlabels=['1 +','2 +'],collabels=['10','20','30'],caption='Broadcast the singleton dimensions. A view may reuse input addresses; output stores must have unique ownership.')
 elif n==9:
  values=[['keep' if j<=i else 'mask' for j in range(4)] for i in range(4)]
  d.matrix(values,colors=[['teal' if j<=i else 'gray' for j in range(4)] for i in range(4)],cell=115,rowlabels=['q0','q1','q2','q3'],collabels=['k0','k1','k2','k3'])
  d.note('Mask scores before softmax','Disallowed scores become negative infinity. Masking probabilities without renormalizing changes the equation.')
 elif n==13:
  d.columns([('A','Rows [1,2] and [3,4].','blue'),('B','Rows [5,6] and [7,8].','violet')])
  d.matrix([[19,22],[43,50]],cell=108,caption='C = A @ B. C[1,0] = 3*5 + 4*7 = 43.')
  d.note('Three independent bounds','Mask M and N output tails and every K input tail. A complete reduction must include each valid K term once.')
 elif n==14:
  d.matrix([[3,-1,2,4,0,0,0,0]],colors=[['teal']*4+['gray']*4],cell=82,caption='Sum over four valid values: 3 - 1 + 2 + 4 = 8. Padded zeros do not change the sum.')
  d.columns([('Maximum','Use negative infinity for invalid lanes. Max is 4.','blue'),('Mean','Divide by 4 valid values, not by the padded extent 8.','violet')])
 elif n==17:
  d.columns([('Product path','x*x contributes x + x = 4 at x=2.','blue'),('Direct path','The added x contributes 1.','teal')]);d.note('At the shared input','Accumulate both paths: dx = 4 + 1 = 5. A backward that overwrites one contribution loses a path.')
 elif n==20:
  d.matrix([[1,2,3,4]],cell=112,caption='Mean = (1+2+3+4)/4 = 2.5.')
  d.note('Variance and normalization','Variance = (2.25+0.25+0.25+2.25)/4 = 1.25. Normalize with (x-2.5)/sqrt(1.25+eps), then apply affine parameters.')
 elif n==21:
  d.flow([('Scores','[ln(2), 0, ln(2)] give relative weights [2,1,2].','blue'),('Online state','With maximum ln(2): denominator 2.5; numerator 7.5 for V=[1,3,5].','violet'),('Output','7.5 / 2.5 = 3. Rescale both running sums whenever the maximum changes.','teal')]);d.note('Storage versus work','Tiling avoids the full score allocation; exact dense attention still evaluates query/key interactions.')
 elif n==22:
  d.matrix([[0,0,20]],cell=132,caption='Call payoff max(S_T-K,0) for each terminal price.')
  d.note('Discount the sample mean','Estimated price = exp(-r*T) * (0+0+20)/3. This three-path arithmetic example does not establish convergence or an error bar.')
 height=d.write();manifest.append(dict(chapter=n,path='docs/images/'+path,title=title,height=height))
for spec in SPECS:make(*spec)
make(0,'From tensor contracts to checked GPU kernels','Read addresses and ownership first; compose, differentiate and measure afterward.', [('Foundations','Types, specialization, grids, strides and masks.','blue'),('Kernels and gradients','Elementwise work, GEMM, reductions and a correct autograd boundary.','violet'),('Applications and evidence','Fused layers, attention and finance; compare independent references before measuring.','teal')])
(ROOT/'tools/illustrations/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Attach figures using chapter numbers, preserving every existing heading.
import re
for p in (ROOT/'docs').glob('part*/*.md'):
 s=p.read_text();m=re.match(r'# Chapter (\d+):',s)
 if not m:continue
 n=int(m[1]);title=SPECS[n-1][1];link=f'../images/ch{n:02}/01-concept.svg'
 marker=f'\n[![{title}]({link})]({link})\n'
 if marker not in s:
  at=s.index('\n');s=s[:at+1]+marker+s[at+1:];p.write_text(s)
p=ROOT/'docs/index.md';s=p.read_text();marker='\n[![From tensor contracts to checked GPU kernels](images/ch00/01-concept.svg)](images/ch00/01-concept.svg)\n'
if marker not in s:
 at=s.index('\n');p.write_text(s[:at+1]+marker+s[at+1:])
print('Created 23 original, font-embedded SVGs and linked all 22 chapters plus Home.')
