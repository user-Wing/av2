"""在单张原图上用原生编码器校准 V1=95 的量化参数。"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from m80_still_compare import frontier, interp
from single_frame import calibration_choices, encoder_command, read_y4m, sha


def estimate(measure, seed):
    """用实测点夹住目标，在 log(bytes) 域内插值。"""
    target = 95
    rows = []
    q = seed
    while True:
        row = measure(q)
        rows.append(row)
        curve = frontier(rows, 'v1')
        match = interp(curve, 'v1', target)
        if match:
            lower = max((r for r in curve if r['v1'] <= target), key=lambda r: r['v1'])
            upper = min((r for r in curve if r['v1'] >= target), key=lambda r: r['v1'])
            lo, hi = sorted((lower['q'], upper['q']))
            if hi - lo > 3 and upper['v1'] - lower['v1'] > .35:
                fraction = (target - lower['v1']) / (upper['v1'] - lower['v1'])
                guess = lower['q'] + fraction * (upper['q'] - lower['q'])
                remaining = set(range(lo + 1, hi)).difference(r['q'] for r in rows)
                if remaining:
                    q = min(remaining, key=lambda v: (abs(v - guess), v))
                    continue
            return dict(bytes95=math.exp(match[0]), neg95=match[1], points=rows)
        direction = -1 if row['v1'] < target else 1
        if len(rows) == 1:
            stride = 6
        else:
            previous = rows[-2]
            slope = (previous['v1'] - row['v1']) / (q - previous['q'])
            stride = max(2, min(24, round(abs(row['v1'] - target) / max(.04, slope)) + 1))
        seen = {r['q'] for r in rows}
        remaining = [v for v in range(q + direction, 0 if direction < 0 else 256, direction) if v not in seen]
        if not remaining:
            return dict(bytes95=None, points=rows)
        q = min(remaining, key=lambda v: abs(v - q - direction * stride))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'encoder', 'scorer', 'v1-model', 'neg-model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--content-type', choices=['animation', 'real'], required=True)
    parser.add_argument('--qp', type=int, choices=range(1, 256), metavar='1..255', default=104)
    parser.add_argument('--rectangles', action='store_true', help='include deep rectangle search with single-pass restoration')
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error('Output must be a new file in an existing directory')
    source, encoder, scorer = (p.resolve(strict=True) for p in (args.input, args.encoder, args.scorer))
    paths = [source, encoder, scorer, args.v1_model, args.neg_model]
    hashes = [sha(p) for p in paths]
    width, height, payload = read_y4m(source)
    candidates = {}
    measured = {}
    env = {k: v for k, v in os.environ.items() if not k.startswith('SVT_AV2_')}
    with tempfile.TemporaryDirectory(prefix='av2-still-calibration-') as directory:
        work = Path(directory)
        (work / 'source.yuv').write_bytes(payload)
        for name, model in [('v1', args.v1_model), ('neg', args.neg_model)]:
            shutil.copyfile(model, work / (name + '.json'))
        for name, settings in calibration_choices(width, args.content_type, rectangles=args.rectangles).items():
            key = tuple(sorted(settings.items()))
            if key not in measured:
                def measure(q):
                    stream, recon = work / 'trial.obu', work / 'trial.yuv'
                    _, _, command = encoder_command(encoder, source, stream, q, args.content_type,
                                                    'auto', width, height, recon, fast_gdf=True)
                    subprocess.run(command, env=dict(env, **settings), check=True, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    subprocess.run([str(scorer), '--reference', 'source.yuv', '--distorted', 'trial.yuv',
                                    '--width', str(width), '--height', str(height), '--pixel_format', '420',
                                    '--bitdepth', '10', '--model', 'path=v1.json:name=v1', '--model',
                                    'path=neg.json:name=neg', '--threads', '1', '--json', '--output', 'scores.json'],
                                   cwd=work, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    scores = json.loads((work / 'scores.json').read_text())['pooled_metrics']
                    return dict(q=q, bytes=stream.stat().st_size, stream_sha256=sha(stream),
                                v1=scores['v1']['mean'], neg=scores['neg']['mean'])
                measured[key] = estimate(measure, args.qp)
            candidates[name] = dict(measured[key], environment=settings)
    eligible = [n for n in candidates if candidates[n]['bytes95'] is not None]
    if not eligible:
        raise ValueError('No candidate brackets V1=95 for this image')
    winner = min(eligible, key=lambda n: candidates[n]['bytes95'])
    if hashes != [sha(p) for p in paths]:
        raise ValueError('Input, encoder, scorer or model changed during calibration')
    report = dict(target_v1=95, source_sha256=hashes[0], encoder_sha256=hashes[1], scorer_sha256=hashes[2],
                  model_sha256={'v1': hashes[3], 'neg': hashes[4]}, content_type=args.content_type,
                  winner=winner, candidates=candidates)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
    print(winner, args.output)


if __name__ == '__main__':
    main()
