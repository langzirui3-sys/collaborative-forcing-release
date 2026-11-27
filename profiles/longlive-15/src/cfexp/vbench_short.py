"""Official full-video VBench scoring, distinct from VBench-Long clip scoring."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

from .vbench_long import (
    OfficialVBenchLongScorer, VBenchLongMeasurement, VBENCH_LONG_DIMENSIONS,
    _canonical_bytes, _sha256_file, _write_bound_json, _require, _detail_score,
    _finite_unit,
)


def validate_short_result(path: Path, staged_videos: dict[str, Path]) -> dict[str, dict[str, float]]:
    """Require exactly one full-video detail per input and dimension."""
    value = json.loads(path.read_text())
    _require(isinstance(value, dict) and set(value) == set(VBENCH_LONG_DIMENSIONS),
             "short VBench dimensions are incomplete")
    expected = {str(p.resolve(strict=True)): key for key, p in staged_videos.items()}
    _require(len(expected) == len(staged_videos) and bool(expected), "duplicate or empty videos")
    scores = {key: {} for key in staged_videos}
    for dimension in VBENCH_LONG_DIMENSIONS:
        item = value[dimension]
        _require(isinstance(item, list) and len(item) == 2 and isinstance(item[1], list),
                 "short VBench detail structure drifted")
        _finite_unit(item[0], dimension)
        seen = set()
        for detail in item[1]:
            _require(isinstance(detail, dict) and isinstance(detail.get('video_path'), str),
                     "invalid short VBench detail")
            # Official custom_input emits full paths; reject split clips and foreign paths.
            reported = Path(detail['video_path'])
            _require(reported.is_absolute() and not reported.is_symlink(), "unbound detail path")
            resolved = str(reported.resolve(strict=True))
            _require(resolved in expected and resolved not in seen, "duplicate or foreign video detail")
            seen.add(resolved)
            score = _detail_score(detail.get('video_results'), label=dimension)
            if dimension == 'imaging_quality':
                score /= 100.0  # Official per-video MUSIQ details are on the 0..100 scale.
            scores[expected[resolved]][dimension] = _finite_unit(score, dimension)
        _require(seen == set(expected), "short VBench video coverage is incomplete")
    return scores


class OfficialVBenchScorer:
    """Reuse audited offline assets/guards, but execute the official short evaluator."""
    def __init__(self, **kwargs):
        self.runtime = OfficialVBenchLongScorer(**kwargs)
        r = self.runtime
        entry, info = r.checkout / 'evaluate.py', r.checkout / 'vbench/VBench_full_info.json'
        _require(not entry.is_symlink() and not info.is_symlink(), "short entry symlink")
        esha, esize = _sha256_file(entry)
        isha, isize = _sha256_file(info)
        self.binding = replace(r.checkout_binding, eval_script=entry, full_info=info,
                               eval_sha256=esha, eval_nbytes=esize,
                               full_info_sha256=isha, full_info_nbytes=isize)

    def _command(self, root: Path) -> list[str]:
        r, b = self.runtime, self.binding
        return [r.python_executable, str(r.launcher), '--benchmark', 'vbench',
                '--asset-manifest', str(r.assets.manifest_path), '--asset-spec', str(r.assets.spec_path),
                '--checkout', str(r.checkout), '--eval-script', str(b.eval_script),
                '--eval-sha256', b.eval_sha256, '--eval-nbytes', str(b.eval_nbytes),
                '--full-info', str(b.full_info), '--full-info-sha256', b.full_info_sha256,
                '--full-info-nbytes', str(b.full_info_nbytes), '--runtime-cache-root', str(r.runtime_cache),
                '--', '--videos_path', str(root / 'videos'), '--dimension', *VBENCH_LONG_DIMENSIONS,
                '--mode', 'custom_input', '--prompt_file', str(root / 'prompts.json'),
                '--load_ckpt_from_local', 'True', '--imaging_quality_preprocessing_mode', 'longer',
                '--output_path', str(root / 'results'), '--full_json_dir', str(b.full_info)]

    def score_videos(self, requests):
        r = self.runtime
        _require(bool(requests), 'empty short VBench request')
        identities, sources = [], {}
        for request in requests:
            key = request.measurement_id
            _require(isinstance(key, str) and re.fullmatch(r'[A-Za-z0-9_-]+', key)
                     and key not in sources, 'invalid or duplicate measurement ID')
            _require(isinstance(request.prompt, str) and bool(request.prompt.strip()), 'empty prompt')
            source = Path(request.video_path).resolve(strict=True)
            _require(source.is_file() and source.is_relative_to(r.run_dir), 'video outside run directory')
            sha, size = _sha256_file(source)
            identities.append({'id': key, 'prompt': request.prompt, 'source': str(source),
                               'sha256': sha, 'nbytes': size})
            sources[key] = source
        identities.sort(key=lambda item: item['id'])
        # Every evaluation attempt is immutable; failed attempts remain available for inspection.
        batch = hashlib.sha256(_canonical_bytes(identities)).hexdigest()
        root = r.run_dir / 'vbench-short' / batch / uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        _require(root.resolve().is_relative_to(r.run_dir), 'short evaluation path escaped run')
        (root / 'videos').mkdir()
        (root / 'results').mkdir()
        staged, prompts = {}, {}
        for item in identities:
            target = root / 'videos' / (item['id'] + '.mp4')
            os.link(sources[item['id']], target)
            staged[item['id']] = target
            prompts[str(target)] = item['prompt']
        _write_bound_json(root / 'prompts.json', prompts)
        command = self._command(root)
        config = {'schema': 'vbench_short_config_v1', 'benchmark': 'VBench', 'mode': 'custom_input',
                  'inputs': identities, 'upstream': self.binding.config_binding(),
                  'offline_assets': r.assets.config_binding(), 'dimensions': list(VBENCH_LONG_DIMENSIONS),
                  'launcher_sha256': r.launcher_sha256, 'command': command,
                  'offline_environment': r.offline_environment,
                  'prompt_file_sha256': _sha256_file(root / 'prompts.json')[0]}
        _write_bound_json(root / 'config.json', config)
        completed = r.runner(command, cwd=str(r.checkout), capture_output=True, text=True,
                             check=False, env=r.environment)
        (root / 'stdout.log').write_text(completed.stdout or '')
        (root / 'stderr.log').write_text(completed.stderr or '')
        _write_bound_json(root / 'exit.json', {'returncode': completed.returncode})
        _require(completed.returncode == 0, 'official short VBench process failed')
        results = list((root / 'results').glob('*_eval_results.json'))
        _require(len(results) == 1 and not results[0].is_symlink(), 'expected one short VBench result')
        scores = validate_short_result(results[0], staged)
        for item in identities:
            _require(_sha256_file(sources[item['id']]) == (item['sha256'], item['nbytes']),
                     'video changed during short VBench evaluation')
        bindings = {}
        for name, path in [('config', root/'config.json'), ('result', results[0]),
                           ('stdout', root/'stdout.log'), ('stderr', root/'stderr.log'),
                           ('exit', root/'exit.json'), ('prompts', root/'prompts.json')]:
            sha, size = _sha256_file(path)
            bindings[name] = {'path': path.relative_to(r.run_dir).as_posix(), 'sha256': sha, 'nbytes': size}
        measurements = {}
        for key, dimensions in scores.items():
            macro = math.fsum(dimensions.values()) / len(dimensions)
            sidecar = root / (key + '.measurement.json')
            _write_bound_json(sidecar, {'schema': 'vbench_short_measurement_v1', 'benchmark': 'VBench',
                              'measurement_id': key, 'bindings': bindings, 'dimension_scores': dimensions,
                              'macro_mean': macro, 'aggregation': 'six_dimension_arithmetic_mean',
                              'claim_scope': 'custom input, not official full benchmark total'})
            measurements[key] = VBenchLongMeasurement(tuple(dimensions.items()), macro,
                                                       dimensions['motion_smoothness'],
                                                       sidecar.relative_to(r.run_dir).as_posix())
        return measurements
