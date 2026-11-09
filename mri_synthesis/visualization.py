"""Generate all selected examples for all six models; inference only."""
import argparse
import json
from pathlib import Path
from mri_synthesis.paths import ROOT, CONFIG_PATH

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from mri_synthesis.data import stack
from mri_synthesis.cli import device_for, load_generator


TITLES = ['Input T1', 'pGAN · 1 slice', 'pGAN · 3 slices',
          'cGAN reg · 1 slice', 'cGAN reg · 3 slices',
          'cGAN unreg · 1 slice', 'cGAN unreg · 3 slices', 'Real T2']
FIGURES = {}


def render(rows, path, title):
    count = len(rows)
    if count not in FIGURES:
        fig, axes = plt.subplots(count, 8, figsize=(16, 2.4*count+0.5), squeeze=False)
        artists = []
        for row in range(count):
            artists.append([])
            for column in range(8):
                ax = axes[row, column]
                artists[row].append(ax.imshow(np.zeros_like(rows[0][1][0]).T, origin='lower',
                                              cmap='gray', vmin=0, vmax=1))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(TITLES[column], fontsize=9)
        fig.subplots_adjust(left=0.035, right=0.995, bottom=0.025,
                            top=0.78 if count == 1 else 0.94, wspace=0.04, hspace=0.12)
        fig.suptitle(title, fontsize=12)
        FIGURES[count] = fig, axes, artists
    fig, axes, artists = FIGURES[count]
    fig._suptitle.set_text(title)
    for row, (z, images) in enumerate(rows):
        for column, image in enumerate(images):
            artists[row][column].set_data(image.T)
        axes[row, 0].set_ylabel(f'Slice {z}', fontsize=9)
    fig.savefig(path, dpi=110, facecolor='white')


def gallery_page(records, sheets):
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MRI synthesis — all examples</title>
<style>body{font:16px system-ui;background:#f5f6fa;color:#172033;margin:24px}
main{max-width:1800px;margin:auto}h1{margin-bottom:8px}select,button{padding:9px;margin:4px;border:1px solid #b6c1d5;border-radius:6px;background:white}
.tools{display:flex;align-items:center;flex-wrap:wrap;gap:10px}img{width:100%;background:white;border-radius:8px}input{flex:1;min-width:160px}
#label{font-weight:600;margin:16px 0}#sheets a{display:inline-block;margin:5px;padding:7px;background:white;border-radius:6px}a{color:#174fb0}p{line-height:1.5}</style>
<main><h1>MRI synthesis: all six models</h1>
<p>Every selected center slice is included. Each row shows input T1, all six generated T2 predictions, and real T2.
All panels use the same [0,1] grayscale scale. Training examples are seen data; validation is development data;
only test subjects are held out for final reporting. This gallery does not retrain models.</p>
<div class="tools"><label>Split <select id="split"><option value="all">All splits</option><option>test</option><option>val</option><option>train</option></select></label>
<label>Subject <select id="subject"></select></label><button id="prev">Previous</button><input id="slider" type="range" min="0" value="0"><button id="next">Next</button></div>
<div id="label"></div><a id="full" target="_blank"><img id="example" alt="T1 and six synthesized T2 predictions beside the real T2 reference"></a>
<p>Click the image to open its full resolution. Use the filters and slider or Previous/Next to browse.</p>
<h2>Contact sheets</h2><p>Each sheet contains five slices. Filter above to narrow this list.</p><div id="sheets"></div></main>
<script>const records=RECORDS, sheets=SHEETS;
const split=document.getElementById('split'), subject=document.getElementById('subject'), slider=document.getElementById('slider');let filtered=[];
function show(){if(!filtered.length)return;let n=Number(slider.value),r=filtered[n];document.getElementById('example').src=r.image;document.getElementById('full').href=r.image;document.getElementById('label').textContent=`${r.split.toUpperCase()} · ${r.subject} · axial slice ${r.slice} · example ${n+1} / ${filtered.length}`;}
function filter(){filtered=records.filter(r=>(split.value==='all'||r.split===split.value)&&(subject.value==='all'||r.subject===subject.value));slider.max=Math.max(0,filtered.length-1);slider.value=0;show();const box=document.getElementById('sheets');box.replaceChildren();for(const s of sheets.filter(r=>(split.value==='all'||r.split===split.value)&&(subject.value==='all'||r.subject===subject.value))){const a=document.createElement('a');a.href=s.image;a.target='_blank';a.textContent=`${s.split} ${s.subject} · sheet ${s.page}`;box.appendChild(a);}}
function subjects(){const old=subject.value;subject.replaceChildren();let a=new Option('All subjects','all');subject.add(a);for(const s of [...new Set(records.filter(r=>split.value==='all'||r.split===split.value).map(r=>r.subject))])subject.add(new Option(s,s));subject.value=[...subject.options].some(o=>o.value===old)?old:'all';filter();}
split.onchange=subjects;subject.onchange=filter;slider.oninput=show;
document.getElementById('prev').onclick=()=>{slider.value=Math.max(0,Number(slider.value)-1);show();};
document.getElementById('next').onclick=()=>{slider.value=Math.min(Number(slider.max),Number(slider.value)+1);show();};
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')document.getElementById('prev').click();if(e.key==='ArrowRight')document.getElementById('next').click();});subjects();</script></html>'''.replace('RECORDS', json.dumps(records)).replace('SHEETS', json.dumps(sheets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['all', 'train', 'val', 'test'], default='all')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = ROOT
    spec = json.loads(CONFIG_PATH.read_text())
    data = root/spec['data']
    manifest = json.loads((data/'manifest.json').read_text())
    device = device_for(args.device)
    models = []
    for item in spec['experiments']:
        model, config = load_generator(root/'runs/quick'/item['name']/'final.pt', device)
        if (config['model'], config['branch'], config['slices']) != (item['model'], item['branch'], item['slices']):
            raise ValueError(f'Unexpected model configuration: {item["name"]}')
        models.append((model, config))
    output = root/'results/quick'/('all_examples' if args.split == 'all' else f'{args.split}_examples')
    output.mkdir(parents=True, exist_ok=True)
    records, sheets = [], []
    splits = ['test', 'val', 'train'] if args.split == 'all' else [args.split]
    for split in splits:
        for subject in manifest['splits'][split]:
            folder = data/split/subject
            x = np.load(folder/'x.npy', mmap_mode='r')
            y = np.load(folder/'y.npy', mmap_mode='r')
            with np.load(folder/'indices.npz') as indices:
                valid = indices['paired']
            predictions = []
            with torch.inference_mode():
                for model, config in models:
                    chunks = []
                    for start in range(0, len(valid), 8):
                        batch = torch.stack([stack(x, int(z), config['slices']) for z in valid[start:start+8]]).to(device)
                        chunks.append(((model(batch)[:, 0].cpu().numpy()+1)/2).clip(0, 1))
                    predictions.append(np.concatenate(chunks))
            destination = output/split/subject
            destination.mkdir(parents=True, exist_ok=True)
            rows = []
            for i, z in enumerate(valid):
                z = int(z)
                images = [x[z]]+[prediction[i] for prediction in predictions]+[y[z]]
                rows.append((z, images))
                path = destination/f'slice_{z:03}.png'
                render([(z, images)], path, f'{split.upper()} · {subject} · axial slice {z}')
                records.append(dict(split=split, subject=subject, slice=z, image=path.relative_to(output).as_posix()))
            for start in range(0, len(rows), 5):
                page = start//5+1
                path = destination/f'contact_sheet_{page}.png'
                render(rows[start:start+5], path, f'{split.upper()} · {subject} · contact sheet {page}')
                sheets.append(dict(split=split, subject=subject, page=page, image=path.relative_to(output).as_posix()))
            print(f'{split}/{subject}: {len(valid)} examples; {len(records)} total', flush=True)
    metadata = dict(examples=records, contact_sheets=sheets,
                    models=[item['name'] for item in spec['experiments']],
                    checkpoint='final.pt', inference_only=True)
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2))
    (output/'index.html').write_text(gallery_page(records, sheets), encoding='utf-8')
    plt.close('all')
    from mlflow.tracking import MlflowClient
    for item in spec['experiments']:
        info_path = root/'runs/quick'/item['name']/'mlflow.json'
        if info_path.exists():
            info = json.loads(info_path.read_text())
            MlflowClient(tracking_uri=info['tracking_uri']).set_tag(info['run_id'], 'all_examples_gallery', str(output/'index.html'))
    print(f'Created {len(records)} examples and {len(sheets)} contact sheets. Open {output/"index.html"}', flush=True)


if __name__ == '__main__':
    main()
