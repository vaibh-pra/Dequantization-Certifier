"""Recompute historical statistics and verify fixed deployment substitution.
No optimization or retraining is performed; saved checkpoints are read only.
"""
from pathlib import Path
import json,sys
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results'/'validated_revision'
OUT.mkdir(exist_ok=True)
names=['quantum','matched','linear','unconstrained']
records={k:[json.loads((ROOT/'results'/'runs'/f'{k}_seed{s}'/'result.json').read_text()) for s in range(3)] for k in names}
reference=np.array([v['final_psnr'] for v in records['quantum']])
rows=[]
labels=['Quantum-parameterised','Ellipse-parameterised','Affine + tanh','MLP + tanh']
tex=[]
for k,label in zip(names,labels):
    ps=np.array([v['final_psnr'] for v in records[k]])
    ss=np.array([v['final_ssim'] for v in records[k]])
    delta=ps-reference; se=delta.std(ddof=1)/np.sqrt(3); half=t.ppf(.975,2)*se
    row=dict(encoder=k,psnr_mean=ps.mean(),psnr_sample_sd=ps.std(ddof=1),ssim_mean=ss.mean(),ssim_sample_sd=ss.std(ddof=1),
             paired_mean=delta.mean(),paired_se=se,ci95=[delta.mean()-half,delta.mean()+half],
             paired_values=delta.tolist(),parameters=records[k][0]['encoder_params'])
    rows.append(row)
    ci='---' if k=='quantum' else f"$[{row['ci95'][0]:+.2f},{row['ci95'][1]:+.2f}]$"
    tex.append(f"{label} & {row['parameters']} & ${ps.mean():.2f} \\pm {ps.std(ddof=1):.2f}$ & ${ss.mean():.4f} \\pm {ss.std(ddof=1):.4f}$ & ${delta.mean():+.2f}$ & {ci} \\\\")
(OUT/'reconstruction_statistics.json').write_text(json.dumps(rows,indent=2))
(ROOT/'paper'/'reconstruction_table.tex').write_text('\\newcommand{\\reconstructionrows}{%\n'+'\n'.join(tex)+'}\n')

import torch
from PIL import Image
from torchvision import transforms
sys.path.insert(0,str(ROOT/'code'))
from celeba_experiment import QuantumEncoder,Decoder,image_to_patch_angles
# A separate direct-coefficient implementation, with no optimizer or learned
# remapping. The source parameters and decoder are held fixed at checkpoint values.
class MappedEncoder(torch.nn.Module):
    def __init__(self,angles):
        super().__init__()
        coeff=[]
        for i in [0,2]:
            a,b=angles[i],angles[i+1]
            coeff.append(torch.stack((torch.cos(b/2)**2,torch.sin(b/2)**2,
                                      -.5*torch.sin(b)*torch.cos(a),.5*torch.sin(b)*torch.cos(a))))
        self.register_buffer('coeff',torch.stack(coeff))
    def forward(self,img):
        x=image_to_patch_angles(img); outs=[]
        for j in range(2):
            u,v=x[2*j:2*j+2]
            features=torch.stack((torch.cos(u),torch.cos(u)*torch.cos(v),torch.sin(u),torch.sin(u)*torch.cos(v)))
            outs.append((self.coeff[j,:,None,:,None,None]*features).sum(dim=0))
        return torch.cat(outs,dim=1)

torch.set_num_threads(1)
files=sorted((ROOT/'lfw_faces'/'test'/'imgs').glob('*.jpg'))[:16]
if len(files)!=16:
    raise RuntimeError('expected 16 held-out local images')
tf=transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor()])
images=torch.stack([tf(Image.open(p).convert('RGB')) for p in files]).double()
sub=[]
for seed in range(3):
    checkpoint=torch.load(ROOT/'results'/'runs'/f'quantum_seed{seed}'/'model.pt',map_location='cpu',weights_only=True)
    encoder=QuantumEncoder(3).double(); encoder.load_state_dict(checkpoint['encoder']);encoder.eval()
    decoder=Decoder(3).double();decoder.load_state_dict(checkpoint['decoder']);decoder.eval()
    mapped=MappedEncoder(encoder.angles.detach()).eval()
    with torch.no_grad():
        a,b=encoder(images),mapped(images)
        ya,yb=decoder(a),decoder(b)
    sub.append(dict(seed=seed,images=16,max_feature_difference=float((a-b).abs().max()),
                    max_pixel_difference=float((ya-yb).abs().max())))
(OUT/'deployment_substitution.json').write_text(json.dumps(dict(rows=sub,files=[str(p.relative_to(ROOT)) for p in files],dtype='float64'),indent=2))
(ROOT/'paper'/'substitution_table.tex').write_text('\\newcommand{\\substitutionrows}{%\n'+'\n'.join(
    f"{r['seed']} & {r['images']} & \\num{{{r['max_feature_difference']:.2e}}} & \\num{{{r['max_pixel_difference']:.2e}}} \\\\" for r in sub)+'}\n')
print(json.dumps(sub,indent=2))
