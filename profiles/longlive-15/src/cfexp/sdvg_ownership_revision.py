"""Narrow source bridge for one infrastructure-failed SDVG shard's sealed repair.

This is not an alternate protocol or permission to select a better rerun.
Only the original, already sealed repair is admitted, with all ordinary row,
history, scoring, device and timing checks still performed by the main reader.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .main_table_merge import _require

BASE = 'ed89a22bbb661682b1be1d843852f3a62f2e729f'
REVISION = '3e64412f0e606bb2e5d9d4fae39ad2510434f738'
MANIFEST = 'experiments/paper128-sdvg-selected-rolling-15s-s1-p1-ntu-v1.manifest.json'
RUN = 'paper128-sdvg-selected-rolling-15s-s1-p1-ntu-v1-20261121T090259Z-3e64412f-a76b15e9-c12bd81c'
ATTEMPT = 'attempt-20261121T090259Z-9cb022e99e0b41e7856e8c4955c7d75a'
INDEX = 'bb0634f94f58de24258e9897db79532770e2ae209f795e6355e97e1db86124bc'
GUARD = 'src/cfexp/main_table_ours_campaign.py'
GUARD_HASHES = {
    BASE: 'e835fe68898e31afa6c56ebd88c3931542066b79755ea29db89f11424d786ea4',
    REVISION: '09c8fb62d3ea0d3646d50c6407e980d301cbb9218bb041b1cb5c724094fe95d4',
}
DIFFERENCES = {
    'src/cfexp/erank_acceptance_pilot.py', 'src/cfexp/insight2_campaign.py',
    'src/cfexp/krea_long_session_gate.py', GUARD,
    'src/cfexp/native_c10_development.py',
    'src/cfexp/paper128_sdvg_selected_report.py', 'src/cfexp/tracked_native_c10.py',
}
PRODUCTION_FILES = (
    'src/cfexp/correction_runtime.py', 'src/cfexp/correction_worker.py',
    'src/cfexp/krea_chunk_backend.py', 'src/cfexp/self_forcing_frame_adapter.py',
    'src/cfexp/main_table_sdvg.py', 'src/cfexp/main_table_sdvg_campaign.py',
    'src/cfexp/tracked_sdvg_selected_main.py', 'src/cfexp/sdvg_accepted_vae.py',
    'src/cfexp/sdvg_vae_transaction.py', 'src/cfexp/accepted_stream.py',
    'src/cfexp/accepted_stream_worker.py', 'src/cfexp/video_encoding_policy.py',
    'src/cfexp/vbench_short.py', 'src/cfexp/vbench_long.py',
    'pyproject.toml', 'uv.lock',
)


def identity(marker, run, manifest, *, model, seconds):
    """Reject any additional run, attempt, population or outcome-based substitute."""
    _require(model == 'rolling' and seconds == 15
        and marker.get('git_sha') == REVISION and marker.get('run_id') == RUN
        and marker.get('attempt_id') == ATTEMPT
        and marker.get('artifact_index_sha256') == INDEX
        and Path(run).name == RUN and manifest == MANIFEST,
        'unapproved SDVG ownership-diagnostic repair identity')


def git_reader(repo):
    env = dict(PATH='/usr/bin:/bin', HOME=os.path.expanduser('~'),
        GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
        GIT_CONFIG_SYSTEM='/dev/null', GIT_NO_REPLACE_OBJECTS='1')
    def git(*args):
        return subprocess.check_output(['git', '--no-replace-objects', '-c',
            'core.hooksPath=/dev/null', '-C', str(repo), *args], env=env)
    return git


def outside_declarations_equal(before, after, *, changed, added=(), docstring=False):
    """Prove imports, constants and other declarations have not changed."""
    old = ast.parse(before); new = ast.parse(after)
    originals = {x.name: x for x in old.body if isinstance(x, (ast.FunctionDef, ast.ClassDef))}
    new.body = [copy.deepcopy(originals[x.name]) if isinstance(x, (ast.FunctionDef, ast.ClassDef))
                and x.name in changed else x for x in new.body
                if not isinstance(x, (ast.FunctionDef, ast.ClassDef)) or x.name not in added]
    if docstring:
        _require(ast.get_docstring(old) is not None and ast.get_docstring(new) is not None,
                 'source module docstring missing')
        new.body[0] = copy.deepcopy(old.body[0])
    _require(ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False),
             'SDVG imported declarations or module initialization changed')


def bridge(repo):
    """Audit the immutable writer sources, never the reader's working-tree code."""
    git = git_reader(repo)
    changes = git('diff', '--name-only', BASE, REVISION, '--',
                  'src/cfexp', 'pyproject.toml', 'uv.lock').decode().splitlines()
    _require(set(changes) == DIFFERENCES, 'unreviewed ownership repair source differences')
    old_manifest = git('show', BASE + ':' + MANIFEST)
    _require(old_manifest == git('show', REVISION + ':' + MANIFEST),
             'repair changed SDVG experiment configuration')
    bound = []
    for item in json.loads(old_manifest)['spec']['inputs']:
        if not item['uri'].startswith('repository://'):
            continue
        path = item['target']; before = git('show', BASE + ':' + path)
        _require(before == git('show', REVISION + ':' + path)
            and hashlib.sha256(before).hexdigest() == item['sha256'],
            'repair changed a frozen SDVG production input')
        bound.append(dict(path=path, sha256=item['sha256']))
    _require(bool(bound), 'repair has no original frozen production inputs')
    operators = []
    for path in PRODUCTION_FILES:
        before=git('show',BASE+':'+path)
        _require(before==git('show',REVISION+':'+path),
                 'repair changed an SDVG generator, scorer, renderer or runtime')
        operators.append(dict(path=path,sha256=hashlib.sha256(before).hexdigest()))
    sources = {}
    for path in (GUARD, 'src/cfexp/insight2_campaign.py', 'src/cfexp/erank_acceptance_pilot.py'):
        before = git('show', BASE + ':' + path); after = git('show', REVISION + ':' + path)
        sources[path] = dict(before_sha256=hashlib.sha256(before).hexdigest(),
                             after_sha256=hashlib.sha256(after).hexdigest())
        if path == GUARD:
            _require(sources[path] == dict(before_sha256=GUARD_HASHES[BASE],
                after_sha256=GUARD_HASHES[REVISION]), 'ownership guard source differs')
            outside_declarations_equal(before, after,
                changed={'_assert_worker_pair_owns_gpus'},
                added={'_ownership_process_record', '_ownership_failure_snapshot'})
        elif path.endswith('insight2_campaign.py'):
            # SDVG imports PromptCase, noise/request protocols and GeneratedChunk,
            # not Insight2Campaign or its separate C10 development constructor.
            outside_declarations_equal(before, after, changed={'Insight2Campaign'},
                added={'validate_prompt_partition'}, docstring=True)
        else:
            # The SDVG campaign imports bound_file, not the eRank entry/CLI.
            outside_declarations_equal(before, after, changed={'run_probe', 'main'})
    return dict(schema='sdvg_ownership_revision_bridge_v1', base_revision=BASE,
        repaired_revision=REVISION, only_approved_repair_run=RUN,
        source_differences=changes, imported_declaration_hashes=sources,
        unchanged_original_input_bindings=bound,
        unchanged_additional_operator_files=operators,
        configuration_unchanged=True, successful_gpu_ownership_predicate_unchanged=True,
        success_queries_per_check=dict(uuid=1,compute_pids=1),
        failure_snapshot_calls_on_success=0,
        successful_guard_python_control_flow_identical=False,
        failure_only_process_metadata_added=True,
        sampler_renderer_scoring_and_runtime_lock_unchanged=True,
        no_retroactive_tracking_or_seal=True, not_whole_source_identity=True)
