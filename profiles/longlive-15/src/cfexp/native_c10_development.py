"""Separate native-four-step development inputs; never substitute for paper128."""
import hashlib
import os
from pathlib import Path

from . import paper_population as population

ARMS = ('rank_burst4_p8_k8', 'rank_burst4_p8_k12', 'rank_burst4_p8_k16')
GATE_REVISION_POLICY = 'same_exact_formal_source_commit'


def validate_options(*, cohort, arm, seconds, sealed_ranks, smoke, env=None):
    env = os.environ if env is None else env
    if (population.enabled(env) or env.get(population.C10_ENV, '0') != '0'
            or any(k in env for k in (population.SEED_ENV, 'CFEXP_PAPER_PROMPT_SHARD_COUNT',
                                      'CFEXP_PAPER_PROMPT_SHARD_INDEX'))
            or cohort != 'main24_burst' or arm not in ARMS or seconds != 30
            or sealed_ranks is not True or smoke is not None):
        raise ValueError('native C10 development requires its separate full-ten-prompt 30s phase')
    for flag in ('CFEXP_SF_SINGLE_GPU', 'CFEXP_SF_JOINT_HANDOFF', 'CFEXP_DRAFT_KV_REUSE',
                 'CFEXP_ACCEPTED_VIDEO_STREAM', 'CFEXP_SDVG_TRACKING',
                 'CFEXP_SDVG_BOUNDED_PROFILE', 'CFEXP_STAGE_TIMING'):
        if env.get(flag) != '1':
            raise ValueError('native C10 development requires single-GPU KV, W&B and bounded profiling')
    for flag in ('CFEXP_SF_TARGET_TWO_STEP', 'CFEXP_SF_TARGET_BURST_INTERIOR_ONE_STEP',
                 'CFEXP_SF_TARGET_BURST_EXIT_TWO_STEP', 'CFEXP_SF_TARGET_LOW_NOISE_REFINEMENT',
                 'CFEXP_ACCEPTED_TARGET_PREFILL'):
        if env.get(flag, '0') != '0':
            raise ValueError('native C10 development forbids shortened Target updates or alternate prefill')
    if env.get('CFEXP_SF_TARGET_EARLY_NATIVE_BURSTS', '0') != '0':
        raise ValueError('native C10 development uses four updates for EVERY Target block')


def select_population(root, protocol):
    root = Path(root).resolve(strict=True)
    definition = population._strict_json(root / population.PROTOCOL)
    if (definition.get('schema') != 'paper_reproduction_population_v1'
            or definition.get('status') != 'frozen'
            or definition.get('source_manifest_sha256') != population.SOURCE_SHA256):
        raise ValueError('original frozen paper population definition differs')
    source_path = root / population.SOURCE_FILE
    if (source_path.is_symlink() or hashlib.sha256(source_path.read_bytes()).hexdigest()
            != population.SOURCE_SHA256):
        raise ValueError('original evaluation population identity differs')
    source = population._strict_json(source_path)
    expected, reference = population._calibration_definition(
        root, definition, [x['prompt'] for x in source['cases']])
    fields = ('prompt_id', 'prompt', 'source_index_zero_based', 'dimensions', 'selection_sha256')
    if [tuple(x[f] for f in fields) for x in protocol.calibration_prompts] != [
            tuple(x[f] for f in fields) for x in expected]:
        raise ValueError('development population differs from the frozen independent C10')
    return ([{'id': x['prompt_id'], 'text': x['prompt']} for x in expected], dict(
        schema='independent_c10_development_binding_v1', calibration_protocol=reference,
        original_evaluation_sha256=population.SOURCE_SHA256, selected_case_count=10,
        selected_prompt_ids=[x['prompt_id'] for x in expected], seed=population.CALIBRATION_SEED,
        disjoint_from_paper128=True, held_out=False, development_only=True,
        may_replace_paper128_results=False))


def reference_population(repo, boundary_config, *, seconds, smoke, environment=None):
    """Bind complete pure controls to the same original development ten cases."""
    env=os.environ if environment is None else environment
    from .krea_timestep_adapter import target_timestep_policy
    from .insight2_run import load_run_config
    from .insight2_protocol import load_protocol
    if (seconds != 30 or smoke is not None or boundary_config is None
            or env.get(population.ENV,'0') != '0' or env.get(population.C10_ENV,'0') != '0'
            or any(k in env for k in (population.SEED_ENV,'CFEXP_PAPER_PROMPT_SHARD_COUNT',
                                      'CFEXP_PAPER_PROMPT_SHARD_INDEX'))):
        raise ValueError('C10 references require the original unsharded ten-case development phase')
    for name in ('CFEXP_SF_SINGLE_GPU','CFEXP_DRAFT_KV_REUSE','CFEXP_ACCEPTED_VIDEO_STREAM',
                 'CFEXP_SDVG_TRACKING','CFEXP_SDVG_BOUNDED_PROFILE','CFEXP_STAGE_TIMING'):
        if env.get(name) != '1':
            raise ValueError('C10 reference requires single-GPU KV, actual tracking and profiling')
    for name in ('CFEXP_SF_TARGET_TWO_STEP','CFEXP_SF_TARGET_BURST_INTERIOR_ONE_STEP',
                 'CFEXP_SF_TARGET_BURST_EXIT_TWO_STEP','CFEXP_SF_TARGET_LOW_NOISE_REFINEMENT',
                 'CFEXP_SF_TARGET_EARLY_NATIVE_BURSTS','CFEXP_ACCEPTED_TARGET_PREFILL',
                 'CFEXP_SF_JOINT_HANDOFF'):
        if env.get(name,'0') != '0':
            raise ValueError('pure C10 reference forbids collaboration or shortened Target updates')
    target_timestep_policy(env)
    config=load_run_config(Path(boundary_config),repo_root=Path(repo))
    return select_population(repo,load_protocol(config.protocol_path))


def validate_development_population(repo, protocol, calibration_prompts, evaluation_prompts):
    from .insight2_campaign import validate_prompt_partition
    selected,binding=select_population(repo,protocol)
    expected=tuple((p['id'],p['text']) for p in selected)
    if (tuple((p.prompt_id,p.prompt) for p in calibration_prompts)!=expected
            or tuple((p.prompt_id,p.prompt) for p in evaluation_prompts)!=expected):
        raise ValueError('development replay differs from the complete original independent C10')
    validate_prompt_partition(calibration_prompts,evaluation_prompts,calibration_replay_development=True)
    return binding


def make_development_campaign(*, repo, protocol, **kwargs):
    """Use the real constructor and preserve original IDs/texts, without held-out claims."""
    from .insight2_campaign import Insight2Campaign
    validate_development_population(repo,protocol,kwargs['calibration_prompts'],kwargs['evaluation_prompts'])
    if tuple(kwargs['seeds'])!=(population.CALIBRATION_SEED,) or kwargs['chunk_count']!=60:
        raise ValueError('development replay requires the frozen C10 seed and sixty chunks')
    return Insight2Campaign(protocol_id='native4-independent-c10-development-v1',
                            calibration_replay_development=True,**kwargs)


def verify_target_gate(run, repo, *, plan_sha256=None):
    """Require the actual sixty-block seal, not the historical two-block check."""
    import json
    from .main_table_merge import _load_frozen_index
    from .metadata import collect_git_metadata
    from .krea_timestep_adapter import target_timestep_policy, target_four_step_schedule, UPSTREAM_TIMESTEP_POLICY
    policy=target_timestep_policy()
    gate_name=('krea-upstream-four-step-native30-profile-ntu-v1' if policy==UPSTREAM_TIMESTEP_POLICY
               else 'krea-long-session-parity-native30-profile-ntu-v1')
    if not run:
        raise ValueError('an original sealed full30s Target parity run is required')
    revision=collect_git_metadata(Path(repo))['sha']
    root=Path(run);queue_binding=None
    if root.is_file():
        if (root.is_symlink() or not isinstance(plan_sha256,str) or len(plan_sha256)!=64
                or hashlib.sha256(root.read_bytes()).hexdigest()!=plan_sha256):
            raise ValueError('native60 gate queue plan hash differs')
        plan=json.loads(root.read_text())
        gates=[t for t in plan.get('tasks',[]) if t.get('manifest')==
               'experiments/'+gate_name+'.manifest.json']
        if (plan.get('schema')!='cfexp_erank_queue_v1' or plan.get('ops')!=str(root.parent)
                or len(gates)!=1 or gates[0].get('commit')!=revision):
            raise ValueError('native60 gate queue source identity differs')
        state_path=root.parent/'state.json'
        if state_path.is_symlink():raise ValueError('unsafe native60 gate queue state')
        state=json.loads(state_path.read_text());row=state.get('tasks',{}).get(gates[0]['id'],{})
        if (state.get('plan_sha256')!=plan_sha256 or row.get('status')!='succeeded'
                or not isinstance(row.get('run_dir'),str) or not Path(row['run_dir']).is_absolute()):
            raise ValueError('native60 gate queue has no original succeeded run')
        queue_binding=dict(plan=str(root),sha256=plan_sha256,task_id=gates[0]['id'])
        root=Path(row['run_dir'])
    frozen, marker = _load_frozen_index(root)
    if marker.get('git_sha') != revision or marker.get('status') != 'SUCCEEDED':
        raise ValueError('wrong or unsuccessful native60 Target gate')
    prefix = 'attempts/' + marker['attempt_id'] + '/'
    path = frozen.file(prefix+'result.json', label='native60 Target gate')
    report = json.loads(path.read_text())
    pairs = report.get('canonical_prefix_comparisons', [])
    if (report.get('schema') != 'krea_long_session_parity_v1'
            or report.get('passed') is not True or report.get('chunk_count') != 60
            or report.get('mismatched_chunks') != [] or report.get('first_mismatch_chunk') is not None
            or len(report.get('requests', [])) != 120
            or report.get('timestep_policy') != policy
            or report.get('denoising_schedule') != list(target_four_step_schedule())
            or [x.get('chunk_index') for x in pairs] != list(range(60))
            or any(x.get('latent_exact') is not True for x in pairs)):
        raise ValueError('incomplete or mismatched sixty-block Target execution')
    resolved = json.loads(frozen.file('resolved-manifest.json', label='gate manifest').read_text())
    manifest = resolved['manifest']
    if (manifest['metadata']['name'] != gate_name
            or target_timestep_policy(manifest['spec']['command'].get('env',{})) != policy):
        raise ValueError('wrong native60 Target gate manifest')
    for item in manifest['spec']['inputs']:
        if item['uri'].startswith('repository://'):
            current = Path(repo)/item['target']
            if current.is_symlink() or hashlib.sha256(current.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Target gate operator source or artifact definition changed')
    return dict(run_id=marker['run_id'], git_sha=marker['git_sha'],
        revision_policy=GATE_REVISION_POLICY,
        queue_binding=queue_binding,
        artifact_index_sha256=marker['artifact_index_sha256'],
        report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), chunk_count=60)
