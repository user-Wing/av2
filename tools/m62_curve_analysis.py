"""Evaluate frozen 2D curves without extrapolating incomplete measurements."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if HERE.name == 'tools' else HERE.parents[1]
sys.path.insert(0, str(REPO / 'tools'))
sys.path.insert(0, str(REPO / 'artifacts/m62-2d-size-20261106/python-deps'))
import joint_quality as jq
import m61_screen as m


def load(path):
    if path.is_dir():
        path = path / 'summary.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        data = data.get('points', data.get('rows', data))
    return [r for r in data if r.get('status', 'PASS_FIXED_DECODE_SCORED') == 'PASS_FIXED_DECODE_SCORED']


def strip(row):
    return {'q': row['job']['q'] if 'job' in row else row['q'], 'bytes': row['bytes'],
            'v1': row['v1'], 'neg': row['neg'], 'stream_sha256': row.get('stream_sha256'),
            'decoded_sha256': row.get('decoded_sha256')}


def aom(resolution):
    data = json.loads((REPO / 'tests/r13/publish/publication/numerical-evidence.json').read_text(encoding='utf-8'))
    return data['groups'][f'animation-{resolution[:-1]}-a']['aom_points']


def coverage(points):
    inside = [p for p in points if 93 <= p['v1'] <= 97]
    return {'points': len(points), 'inside_93_97': len(inside),
            'below_93': bool(points) and min(p['v1'] for p in points) < 93,
            'above_97': bool(points) and max(p['v1'] for p in points) > 97,
            'eligible': len(inside) >= 5 and bool(points) and min(p['v1'] for p in points) < 93 and max(p['v1'] for p in points) > 97}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--resolution', choices=['1080p', '2160p'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source_points = [p for p in load(args.candidate)
                     if 'job' not in p or p['job']['source']['group'] == args.resolution + '-2D']
    assert all('job' not in p or (p['job']['source']['width'], p['job']['source']['height']) ==
               ((1920, 1080) if args.resolution == '1080p' else (3840, 2160)) for p in source_points)
    points = [strip(p) for p in source_points]
    if args.candidate.name == 'summary.json' or args.candidate.is_dir():
        for row, original in zip(points, source_points):
            directory = args.candidate if args.candidate.is_dir() else args.candidate.parent
            point = directory / 'points' / original['job']['id']
            if original.get('stream_sha256'):
                assert m.sha(point / 'stream.obu') == original['stream_sha256']
    assert points, 'No completed candidate points for requested resolution'
    result = {'candidate': str(args.candidate.resolve()), 'points': points,
              'coverage': coverage(points), 'aom_coverage': coverage(aom(args.resolution)),
              'scope': 'Development 2D-01; full coverage requires complete points, identity audit and local QC.',
              'comparisons': {}}
    for method in ['pchip', 'linear']:
        for constraint in [('v1',), ('v1', 'neg')]:
            label = 'primary_v1' if constraint == ('v1',) else 'dual'
            comparison = jq.compare_joint(points, aom(args.resolution), method, constraints=constraint)
            if 'samples' in comparison:
                del comparison['samples']
            result['comparisons'][f'{label}_{method}'] = comparison
    m.save(args.out, result)
    for key, value in result['comparisons'].items():
        print(key, 'coverage=', value['full_coverage'], 'ratio=', value.get('mean_ratio'),
              'worst=', value.get('worst_sample', {}).get('ratio'))
    print('coverage', result['coverage'], 'AOM', result['aom_coverage'])


if __name__ == '__main__':
    main()
