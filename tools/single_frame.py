"""严格单帧 Y4M 分类与编码；完成执行不表示通过质量验收。"""
import argparse
from array import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_y4m(path):
    with Path(path).open('rb') as stream:
        header = stream.readline(4097)
        if len(header) > 4096 or not header.endswith(b'\n'):
            raise ValueError('Invalid Y4M header length/termination')
        fields = header.decode('ascii').strip().split()
        if not fields or fields.pop(0) != 'YUV4MPEG2':
            raise ValueError('Expected YUV4MPEG2')
        tags = {}
        for field in fields:
            key, value = field[0], field[1:]
            if key not in 'WHFCIAX' or (key != 'X' and key in tags):
                raise ValueError('Unknown/duplicate Y4M field')
            tags[key] = value
        if (not re.fullmatch(r'[0-9]+', tags.get('W', '')) or
                not re.fullmatch(r'[0-9]+', tags.get('H', ''))):
            raise ValueError('Invalid Y4M dimensions')
        width, height = int(tags['W']), int(tags['H'])
        if (width, height) not in ((1920, 1080), (3840, 2160)) or tags.get('C') != '420p10':
            raise ValueError('Only 1920x1080/3840x2160 C420p10 is supported')
        if tags.get('I', 'p') != 'p' or not re.fullmatch(r'[1-9][0-9]*:[1-9][0-9]*', tags.get('F', '')):
            raise ValueError('Expected progressive input with a positive frame rate')
        if stream.readline(4097) != b'FRAME\n':
            raise ValueError('Expected an untagged FRAME line')
        payload = stream.read(width * height * 3 + 1)
        if len(payload) != width * height * 3:
            raise ValueError('Expected exactly one complete frame and no trailing data')
    samples = array('H', payload)
    if sys.byteorder != 'little':
        samples.byteswap()
    if max(samples) > 1023:
        raise ValueError('Payload contains samples outside 10-bit range')
    return width, height, payload


def preview_command(ffmpeg, width, height, raw, jpeg):
    return [str(ffmpeg), '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '1',
            '-f', 'rawvideo', '-pixel_format', 'yuv420p10le', '-video_size', f'{width}x{height}',
            '-framerate', '1', '-i', str(raw), '-frames:v', '1', '-filter_threads', '1',
            '-vf', 'scale=640:-2', '-q:v', '3', '-threads', '1', str(jpeg)]


def encoder_command(encoder, source, output, qp, content, profile, width, height, recon=None, *, fast_gdf=False):
    if (width, height) not in ((1920, 1080), (3840, 2160)):
        raise ValueError('Unsupported profile dimensions')
    selected = ('animation' if content == 'animation' and width == 1920 else 'photo') if profile == 'auto' else profile
    model = {'animation': 'avm-ai', 'photo': 'avm-ai150'}[selected]
    command = [str(encoder), '--input', str(source), '--output', str(output), '--still-picture',
               '--threads', '1', '--qp', str(qp), '--intra-select', 'cfl', '--partition-select', 'mixed8',
               '--rd-mode', 'fast', '--entropy-adapt', 'on', '--transform-select', 'full',
               '--coeff-opt', 'rdoq', '--angle-search', 'rd', '--mrl-search', 'rd', '--tree-refine', 'screen' if fast_gdf else 'wide',
               '--cfl-search', 'rd', '--mhccp-search', 'rd', '--rd-lambda', model, '--content-type', content]
    if recon is not None:
        command += ['--export-recon', str(recon)]
    return selected, model, command


def quality_settings(width, content, *, rounding):
    matrix = content == 'animation' or width == 1920
    settings = dict(SVT_AV2_GDF='still-quality', SVT_AV2_RECT_RDOQ='1', SVT_AV2_IBP='1',
                    SVT_AV2_STILL_EDGE_FILTER='1', SVT_AV2_STILL_UV_RD='0', SVT_AV2_INTRA_IST='off',
                    SVT_AV2_SSIM_RD='off', SVT_AV2_LUMA_COMPENSATION='off', SVT_AV2_DCT64_MATCH='key',
                    SVT_AV2_QM12='all' if matrix else 'off', SVT_AV2_UV_DELTA_Q='-23')
    if matrix:
        settings['SVT_AV2_QM_LEVEL'] = '4' if width == 1920 else '8'
    if rounding:
        settings['SVT_AV2_FP_ROUND'] = 'matrix'
    return settings


def deep_matrix_settings(width, content):
    settings = quality_settings(width, content, rounding=True)
    screened = width == 3840 and content == 'real'
    settings.update(SVT_AV2_QM12='all', SVT_AV2_QM_LEVEL='12' if screened else '8')
    if not screened:
        settings.update(SVT_AV2_STILL_RECT_RD='deep', SVT_AV2_UV_DELTA_Q='-8')
    return settings, screened


def calibration_choices(width, content, *, rectangles=False):
    base = quality_settings(width, content, rounding=True)
    choices = {'quality': quality_settings(width, content, rounding=False), 'round': base}
    for name, level, delta in [('matrix4', 4, -23), ('matrix8', 8, -23), ('matrix12', 12, -23),
                               ('uv8', None, -8), ('uv0', None, 0), ('uv8m8', 8, -8)]:
        settings = dict(base, SVT_AV2_UV_DELTA_Q=str(delta))
        if level is not None:
            settings.update(SVT_AV2_QM12='all', SVT_AV2_QM_LEVEL=str(level))
        choices[name] = settings
    if rectangles:
        for name in ('uv0', 'uv8m8'):
            choices['rect-' + name] = dict(choices[name], SVT_AV2_STILL_SEARCH='rect', SVT_AV2_STILL_RECT_RD='deep')
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output', 'encoder'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--qp', type=int, required=True)
    parser.add_argument('--tile-columns', type=int, choices=(1, 2, 4, 8), help='uniform columns for parallel still encoding')
    parser.add_argument('--threads', type=int, choices=range(1, 65), metavar='1..64')
    parser.add_argument('--recon', type=Path)
    parser.add_argument('--content-type', choices=('auto', 'animation', 'real'), default='auto')
    parser.add_argument('--profile', choices=('auto', 'animation', 'photo'), default='auto')
    search = parser.add_mutually_exclusive_group()
    search.add_argument('--content-matrix', action='store_true', help='deep rectangles on animation, screened QM4/QM12 on HD/UHD live action')
    search.add_argument('--deep-matrix', action='store_true', help='deep rectangle search with QM8, or screened QM12 for 2160p live action')
    search.add_argument('--directional-matrix', action='store_true', help='directional tree candidate and unit-level PC-Wiener, four columns and four threads by default')
    search.add_argument('--balanced-matrix', action='store_true', help='limit partition, angle and reference-line search in the matrix profile')
    search.add_argument('--quality-calibration', type=Path, help='apply the calibration for this source frame')
    search.add_argument('--quality-matrix', action='store_true', help='use still-quality GDF and frequency-specific quantizer rounding')
    search.add_argument('--quality-gdf', action='store_true', help='use still-picture GDF model offset 3, resolution-based strength and UV delta -23')
    search.add_argument('--fusion-gdf', action='store_true', help='use screened trees, resolution-based matrices and UV delta -8')
    search.add_argument('--fast-gdf', action='store_true', help='use screened mixed8 trees and single-pass restoration')
    search.add_argument('--deep-gdf', nargs='?', const='base', choices=('base', 'qm12'),
                        help='use deep rectangle RD, rectangle RDOQ, IBP, edge filtering and KEY GDF, optionally with QM12')
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--classifier-dir', type=Path, default=Path(__file__).parent / 'public-classifier')
    args = parser.parse_args()
    args.tile_columns = args.tile_columns or (4 if args.directional_matrix else 1)
    args.threads = args.threads or (args.tile_columns if args.directional_matrix else 1)
    if not 1 <= args.qp <= 255:
        parser.error('QP must be in [1,255]')
    if args.content_type == 'auto' and (args.ffmpeg is None or args.checkpoint is None):
        parser.error('Automatic classification requires explicit --ffmpeg and --checkpoint paths')
    source, output, encoder = args.input.resolve(strict=True), args.output.resolve(), args.encoder.resolve(strict=True)
    report_path, log_path = Path(str(output) + '.json'), Path(str(output) + '.log')
    if any(p.exists() for p in (output, report_path, log_path)) or not output.parent.is_dir():
        raise ValueError('Output/JSON/log must be fresh files in an existing directory')
    recon = args.recon.resolve() if args.recon is not None else None
    if recon is not None and (recon.exists() or not recon.parent.is_dir() or recon in (output, report_path, log_path)):
        raise ValueError('Reconstruction must be a separate fresh file in an existing directory')
    width, height, payload = read_y4m(source)
    report = dict(status='STARTED', input=str(source), input_sha256=sha(source), dimensions=[width, height],
                  encoder=str(encoder), encoder_sha256=sha(encoder), qp=args.qp, requested_profile=args.profile,
                  requested_content_type=args.content_type, quality_acceptance='NOT_MEASURED', encoding_input_resized=False,
                  wrapper_sha256=sha(__file__), profile_policy='r13-content-resolution')
    if recon is not None:
        report['recon'] = str(recon)
    env = {k: v for k, v in os.environ.items() if not k.startswith('SVT_AV2_TEST_') and k != 'SVT_AV2_M6_DIP'}
    balanced = args.balanced_matrix or args.directional_matrix
    quality = args.content_matrix or args.deep_matrix or args.quality_gdf or args.quality_matrix or balanced or args.quality_calibration is not None
    matrix_profile = args.fusion_gdf or quality
    screened = args.fast_gdf or matrix_profile
    if quality:
        env = {k: v for k, v in env.items() if not k.startswith('SVT_AV2_')}
    elif args.deep_gdf or screened:
        env = {k: v for k, v in env.items() if not k.startswith('SVT_AV2_')}
        settings = dict(SVT_AV2_STILL_RECT_RD='deep', SVT_AV2_RECT_RDOQ='1', SVT_AV2_IBP='1',
                        SVT_AV2_STILL_EDGE_FILTER='1', SVT_AV2_GDF='key', SVT_AV2_STILL_UV_RD='0',
                        SVT_AV2_INTRA_IST='off', SVT_AV2_QM12='all' if args.deep_gdf == 'qm12' else 'off',
                        SVT_AV2_SSIM_RD='off',
                        SVT_AV2_LUMA_COMPENSATION='off')
        if screened:
            settings.pop('SVT_AV2_STILL_RECT_RD')
            settings.update(SVT_AV2_DCT64_MATCH='key')
        env.update(settings)
        report['tool_environment'] = settings
    try:
        content = args.content_type
        if content == 'auto':
            ffmpeg = args.ffmpeg.resolve(strict=True)
            package = args.classifier_dir.resolve(strict=True)
            spec = importlib.util.spec_from_file_location('frozen_predict', package / 'predict.py')
            predictor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(predictor)
            runtime = predictor.Runtime(args.checkpoint)
            with tempfile.TemporaryDirectory(prefix='single-frame-preview-') as directory:
                raw, jpeg = Path(directory) / 'frame.yuv', Path(directory) / 'preview.jpg'
                raw.write_bytes(payload)
                command = preview_command(ffmpeg, width, height, raw, jpeg)
                subprocess.run(command, check=True, env=env)
                report['classification'] = runtime.infer_bytes(jpeg.read_bytes())
                report['preview'] = dict(command=command, sha256=sha(jpeg), ffmpeg_sha256=sha(ffmpeg))
            report['classifier'] = {name: sha(package / name) for name in ('predict.py', 'runtime.json')}
            report['checkpoint_sha256'] = runtime.config['model']['sha256']
            content = report['classification']['encoder_content_type']
        if quality:
            settings = quality_settings(width, content, rounding=args.deep_matrix or args.quality_matrix or balanced)
            if args.content_matrix and content == 'real' and width == 1920:
                settings = quality_settings(width, content, rounding=True)
            elif args.deep_matrix or args.content_matrix:
                settings, screened = deep_matrix_settings(width, content)
            if balanced:
                settings['SVT_AV2_STILL_SEARCH'] = 'balanced'
                if width == 3840 and content == 'real':
                    settings.update(SVT_AV2_QM12='all', SVT_AV2_QM_LEVEL='4')
            if args.directional_matrix:
                settings.update(SVT_AV2_STILL_SEARCH='directional', SVT_AV2_TILE_RESTORATION='pcw')
            if args.quality_calibration:
                calibration = json.loads(args.quality_calibration.read_text(encoding='utf-8-sig'))
                if calibration['source_sha256'] != report['input_sha256'] or args.profile != 'auto':
                    raise ValueError('Calibration requires the same input frame and the automatic RD profile')
                if calibration.get('content_type', content) != content:
                    raise ValueError('Calibration requires the content type used for the pilot encodes')
                winner = calibration['winner']
                settings = calibration_choices(width, content, rectangles=True)[winner]
                if settings != calibration['candidates'][winner]['environment']:
                    raise ValueError('Calibration settings disagree with this content type or profile')
                report['calibration'] = dict(path=str(args.quality_calibration.resolve()), winner=winner)
            env.update(settings)
            report['tool_environment'] = settings
        elif screened:
            matrix = content == 'animation' or (matrix_profile and width == 1920)
            settings['SVT_AV2_QM12'] = 'all' if matrix else 'off'
            if matrix_profile:
                settings['SVT_AV2_UV_DELTA_Q'] = '-8'
                if matrix:
                    settings['SVT_AV2_QM_LEVEL'] = '4' if width == 1920 else '8'
            env.update(settings)
        selected, model, command = encoder_command(encoder, source, output, args.qp, content, args.profile, width, height, recon, fast_gdf=screened)
        command[command.index('--threads') + 1] = str(args.threads)
        if args.tile_columns > 1:
            command.extend(['--tile-columns', str(args.tile_columns)])
        report.update(content_type=content, selected_profile=selected, rd_lambda=model, command=command)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        with log_path.open('xb') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        report['encoder_exit_code'] = process.returncode
        process.check_returncode()
        if (sha(source) != report['input_sha256'] or sha(encoder) != report['encoder_sha256'] or
                sha(__file__) != report['wrapper_sha256']):
            raise ValueError('Input, encoder or wrapper changed during execution')
        if not output.is_file() or output.stat().st_size == 0:
            raise ValueError('Encoder produced no nonempty output')
        if recon is not None:
            if not recon.is_file():
                raise ValueError('Encoder produced no requested reconstruction')
            report.update(recon_sha256=sha(recon), recon_bytes=recon.stat().st_size)
        report.update(status='ENCODE_COMPLETED_NOT_QUALITY_VERIFIED', output=str(output),
                      output_sha256=sha(output), output_bytes=output.stat().st_size)
    except Exception as error:
        report.update(status='FAILED', error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(str(report_path))


if __name__ == '__main__':
    main()
