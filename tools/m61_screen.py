"""Resumable M6.1 r2 screening; results are development evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from single_frame import encoder_command, read_y4m

REPO = Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((REPO/'tests/r13/protocol.json').read_text(encoding='utf-8'))
    groups = ['1080p-2D', '1080p-3D', '2160p-2D', '2160p-3D', '2160p-Real']
    sources = []
    original = {r['file']:r for r in json.loads((REPO/'tests/Sources/Intra/manifest.json').read_text(encoding='utf-8'))}
    for number in range(1, 4):
        for group in groups:
            path = REPO/f'tests/Sources/Intra/{group}-{number:02}.y4m'
            width, height, _ = read_y4m(path)
            digest = sha(path)
            assert digest == original[path.name]['sha256'], path
            sources.append(dict(path=str(path), sha256=digest, width=width, height=height,
                                content='real' if 'Real' in group else 'animation', group=group))
    programs = {name:Path(protocol['tools'][name]['path']) for name in ['aomenc','aomdec','avmdec','scorer']}
    programs['baseline'] = REPO/'build/git-sync-check-r2/bin/SvtAv2EncApp.exe'
    programs['lambda125'] = REPO/'artifacts/m61-r1-lambda125/build/bin/SvtAv2EncApp.exe'
    frozen = {}
    for name, path in programs.items():
        digest = sha(path)
        if name in protocol['tools']:
            assert digest == protocol['tools'][name]['sha256'], name
        target = out/(name+'.exe')
        shutil.copyfile(path, target)
        frozen[name] = dict(path=str(target), sha256=digest)
    jobs = []
    for index in range(7):
        for source in sources:
            for profile in ['aom','baseline','lambda125','alternate']:
                q = [8,12,16,20,24,28,36][index] if profile=='aom' else [72,84,92,100,108,116,132][index]
                jobs.append(dict(id=f"{Path(source['path']).stem}-{profile}-q{q}", source=source, profile=profile, q=q))
    config = dict(status='PREPARED', phase='SCREEN_15_DEVELOPMENT_FRAMES', jobs=jobs,
                  programs=frozen, models=protocol['models'], aom_flags=protocol['aom_flags'],
                  runner_sha256=sha(__file__), source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  source_manifest={p:sha(REPO/p) for p in subprocess.check_output(['git','ls-files','Source','include','app','CMakeLists.txt'],cwd=REPO,text=True).splitlines()},
                  sample_policy='First three per weak group; last 20% per group reserved for independent validation. Remaining sources available for expansion.',
                  performance_scope='Concurrent screening times are diagnostic; formal matched-quality serial repeat timing required.')
    save(out/'campaign.json', config)
    shutil.copyfile(__file__, out/'runner-frozen.py')
    return config

def execute(job, config, out):
    directory = out/'points'/job['id']
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/'result.json'
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding='utf-8'))
        if old['status']=='PASS_FIXED_DECODE_SCORED':
            assert sha(directory/'stream.obu')==old['stream_sha256']
            return old
        return old  # Failed attempts remain visible; do not silently overwrite.
    source, profile, q = job['source'], job['profile'], job['q']
    result = dict(job=job, status='RUNNING', quality_acceptance='SCREEN_ONLY', commands=[])
    save(result_path, result)
    env = {k:v for k,v in os.environ.items() if not k.startswith('SVT_AV2_TEST_') and k!='SVT_AV2_M6_DIP'}
    def run(command, label):
        started=time.perf_counter()
        entry=dict(command=list(map(str,command)), label=label)
        result['commands'].append(entry)
        with (directory/(label+'.log')).open('wb') as log:
            process=subprocess.run(entry['command'],cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=config.get('timeout_seconds',10800))
        entry.update(exit_code=process.returncode, diagnostic_seconds=time.perf_counter()-started)
        process.check_returncode()
    try:
        assert sha(source['path'])==source['sha256']
        w,h,payload=read_y4m(source['path'])
        (directory/'reference.yuv').write_bytes(payload)
        stream=directory/'stream.obu'
        if profile=='aom':
            command=[config['programs']['aomenc']['path'],*config['aom_flags'],f'--cq-level={q}','-o',str(stream),source['path']]
            decoder='aomdec'
        else:
            encoder=config['programs']['lambda125' if profile=='lambda125' else 'baseline']['path']
            selected='auto' if profile=='baseline' else 'animation' if profile=='lambda125' else ('photo' if source['content']=='animation' and w==1920 else 'animation')
            fast_gdf=config.get('fast_gdf',False)
            if fast_gdf:
                env['SVT_AV2_QM12']='all' if source['content']=='animation' or env.get('SVT_AV2_QM_LEVEL') else 'off'
                result['tool_environment']={k:v for k,v in env.items() if k.startswith('SVT_AV2_')}
            _,_,command=encoder_command(encoder,source['path'],stream,q,source['content'],selected,w,h,directory/'internal.yuv',fast_gdf=fast_gdf)
            decoder='avmdec'
        run(command,'encode')
        run([config['programs'][decoder]['path'],'--rawvideo','--output-bit-depth=10','-o',str(directory/'decoded.yuv'),str(stream)],'decode')
        assert (directory/'decoded.yuv').stat().st_size==w*h*3
        if profile!='aom':
            assert sha(directory/'decoded.yuv')==sha(directory/'internal.yuv'), 'Reconstruction mismatch'
        models=config['models']['1080' if w==1920 else '2160']
        for metric in ('v1','neg'):
            assert sha(models[metric]['path'])==models[metric]['sha256']
            shutil.copyfile(models[metric]['path'],directory/(metric+'.json'))
        run([config['programs']['scorer']['path'],'--reference','reference.yuv','--distorted','decoded.yuv','--width',str(w),'--height',str(h),'--pixel_format','420','--bitdepth','10','--model','path=v1.json:name=v1','--model','path=neg.json:name=neg','--threads','1','--json','--output','metrics.json'],'score')
        metrics=json.loads((directory/'metrics.json').read_text())['pooled_metrics']
        result.update(status='PASS_FIXED_DECODE_SCORED',bytes=stream.stat().st_size,v1=metrics['v1']['mean'],neg=metrics['neg']['mean'],stream_sha256=sha(stream),decoded_sha256=sha(directory/'decoded.yuv'),models=models)
        assert sha(source['path'])==source['sha256']
        save(result_path,result)
        for name in ['reference.yuv','decoded.yuv','internal.yuv']:
            (directory/name).unlink(missing_ok=True)  # Only this job's regenerable scratch pixels.
    except Exception as error:
        result.update(status='FAILED',error=repr(error))
        save(result_path,result)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    args=parser.parse_args()
    out=args.out.resolve()
    config=prepare(out) if args.prepare else json.loads((out/'campaign.json').read_text(encoding='utf-8'))
    if args.prepare:
        print('Prepared',len(config['jobs']),'jobs'); return
    assert sha(__file__)==config['runner_sha256'], 'Runner changed; use frozen runner'
    for program in config['programs'].values(): assert sha(program['path'])==program['sha256']
    counts={}
    save(out/'status.json',dict(status='RUNNING',pid=os.getpid(),total=len(config['jobs']),completed=0))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,j,config,out) for j in config['jobs']]
        for future in as_completed(futures):
            result=future.result()
            counts[result['status']]=counts.get(result['status'],0)+1
            save(out/'status.json',dict(status='RUNNING',pid=os.getpid(),total=len(futures),completed=sum(counts.values()),counts=counts,last=result['job']['id']))
    save(out/'status.json',dict(status='SCREEN_COMPLETED_WITH_FAILURES' if counts.get('FAILED') else 'SCREEN_COMPLETED_NOT_ACCEPTED',total=len(config['jobs']),counts=counts))

if __name__=='__main__': main()
