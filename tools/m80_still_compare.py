"""Compare M80 points on the measured log-rate/quality envelope only."""
import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from m61_screen import save, sha


def read_point(path: Path, *, with_config=False) -> dict | None:
    row = json.loads(path.read_text(encoding='utf-8'))
    if row['status'] not in ('SCORED', 'PASS_FIXED_DECODE_SCORED'):
        return None
    if row.get('fixed_compatible') is False:
        raise ValueError(f'fixed AV2 decoder mismatch: {path}')
    stream = path.parent / 'stream.obu'
    if stream.stat().st_size != row['bytes'] or sha(stream) != row['stream_sha256']:
        raise ValueError(f'bitstream identity mismatch: {path}')
    if 'job' in row:
        job = row['job']
        camp = json.loads((path.parents[2] / 'campaign.json').read_text(encoding='utf-8'))
        row.update(source=job['source'], q=job['q'], models=camp['models'][str(job['source']['height'])])
    point = {k: row[k] for k in ('source', 'models', 'q', 'bytes', 'v1', 'neg', 'stream_sha256')}
    if with_config:
        encode = next(command for command in row['commands'] if command['label'] == 'encode')
        if 'job' in row:
            encoder = camp['programs']['aomenc' if job['profile'] == 'aom' else job['profile']]['sha256']
            args = encode['command'][1:]
            environment = path.parents[2] / 'environment.json'
            # Without an environment manifest only points from this one campaign can be combined.
            settings = (json.loads(environment.read_text(encoding='utf-8-sig')) if environment.exists()
                        else {'campaign': str(path.parents[2].resolve())})
            if 'tool_environment' in row:
                settings = row['tool_environment']
        else:
            encoder = row['tools']['aomenc' if row['variant'] == 'aom-p0' else 'new_avmenc']['sha256']
            args = encode['argv'][1:]
            settings = {'variant': row['variant']}
        flags = []
        skip_value = False
        for arg in args:
            if skip_value:
                skip_value = False
            elif arg in ('--input', '--output', '--export-recon', '--qp', '-o'):
                skip_value = True
            elif not arg.startswith(('--qp=', '--cq-level=', '--recon=')) and arg != row['source']['path']:
                flags.append(arg)
        point['config'] = {'encoder_sha256': encoder, 'flags': flags, 'settings': settings}
    return point


def frontier(rows: list[dict], metric: str) -> list[dict]:
    """Discard dominated points; a better metric point never costs fewer bytes on the envelope."""
    result = []
    for row in sorted(rows, key=lambda a: (a['bytes'], -a[metric])):
        if not result or row[metric] > result[-1][metric]:
            result.append(row)
    return result


def interp(curve: list[dict], metric: str, x: float):
    other = 'neg' if metric == 'v1' else 'v1'
    if not curve or not curve[0][metric] <= x <= curve[-1][metric]:
        return None
    for a in curve:
        if a[metric] == x:
            return math.log(a['bytes']), a[other]
    for a, b in zip(curve, curve[1:]):
        if a[metric] < x < b[metric]:
            t = (x - a[metric]) / (b[metric] - a[metric])
            return math.log(a['bytes']) * (1-t) + math.log(b['bytes']) * t, a[other] * (1-t) + b[other] * t
    raise ValueError(f'no interpolation interval for {metric} {x}')


def summarize(candidate, reference, metric, requested):
    x, y = frontier(candidate, metric), frontier(reference, metric)
    if not x or not y:
        return None
    other = 'neg' if metric == 'v1' else 'v1'
    points = []
    for row in candidate:
        match = interp(y, metric, row[metric])
        if match:
            points.append(dict(q=row['q'], target=row[metric], ratio=row['bytes'] / math.exp(match[0]),
                               other_score_delta=row[other] - match[1]))
    lo = max(x[0][metric], y[0][metric], requested[0])
    hi = min(x[-1][metric], y[-1][metric], requested[1])
    result = dict(point_ratios=points, candidate_frontier_q=[r['q'] for r in x],
                  reference_frontier_q=[r['q'] for r in y])
    if lo < hi:
        grid = sorted({lo, hi, *(r[metric] for r in x+y if lo < r[metric] < hi)})
        vals = []
        for q in grid:
            a, b = interp(x, metric, q), interp(y, metric, q)
            vals.append((a[0]-b[0], a[1]-b[1]))
        average = [sum((r-l) * fmean((a[i], b[i])) for l, r, a, b in
                       zip(grid, grid[1:], vals, vals[1:])) / (hi-lo) for i in range(2)]
        result.update(common_interval=[lo, hi], geometric_mean_ratio=math.exp(average[0]),
                      best_ratio=math.exp(min(a for a, _ in vals)), worst_ratio=math.exp(max(a for a, _ in vals)),
                      mean_other_score_delta=average[1])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', nargs='+', required=True, type=Path)
    p.add_argument('--candidate', nargs='+', required=True, type=Path)
    p.add_argument('--full-band', action='store_true',
                   help='also check 93..97 density and evaluate PCHIP/linear single and joint constraints')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    reference = [r for path in a.reference if (r := read_point(path, with_config=a.full_band)) is not None]
    candidate = [r for path in a.candidate if (r := read_point(path, with_config=a.full_band)) is not None]
    if not candidate or not reference:
        p.error('need completed candidate and reference points')
    identities = {(r['source']['sha256'], tuple(r['models'][m]['sha256'] for m in ('v1', 'neg')))
                  for r in candidate + reference}
    if len(identities) != 1:
        raise ValueError('different source or models')
    result = dict(scope='DIAGNOSTIC_NO_EXTRAPOLATION', candidate=candidate, reference=reference,
                  v1=summarize(candidate, reference, 'v1', (93, 97)),
                  neg=summarize(candidate, reference, 'neg', (-math.inf, math.inf)))
    if a.full_band:
        from m62_curve_analysis import coverage, jq
        band = dict(candidate=coverage(candidate), reference=coverage(reference))
        for label, points in (('candidate', candidate), ('reference', reference)):
            configs = [point.pop('config') for point in points]
            if any(config != configs[0] for config in configs):
                raise ValueError(f'{label} points use different encoder binaries or settings')
            if len({p['q'] for p in points}) != len(points):
                raise ValueError(f'{label} contains repeated quantizer settings')
            band[label]['config'] = configs[0]
        result['full_band'] = band
        if band['candidate']['eligible'] and band['reference']['eligible']:
            for method in ('pchip', 'linear'):
                for name, metrics in (('primary_v1', ('v1',)), ('dual', ('v1', 'neg'))):
                    comparison = jq.compare_joint(candidate, reference, method, constraints=metrics)
                    comparison.pop('samples', None)
                    band[f'{name}_{method}'] = comparison
    save(a.out, result)


if __name__ == '__main__':
    main()
