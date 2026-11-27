"""Verify the coupled LongLive/Rolling VAE; never install generation KV code.

Both frozen upstream archives contain the same VAE implementation and weights.
Their wrappers are independently hash-bound before parsing normalization and
checking the native decode method. No upstream wrapper is executed here.
"""
import ast
import hashlib
import json
import os
import tarfile

from .streaming_vae_probe import VAE_SHA, WEIGHTS_SHA

ENV = 'CFEXP_NATIVE_VIDEO_STREAM_MODEL'
SERIAL_READY_SCOPE = 'native_warm_models_cached_text_serial_video_ready_v1'
OVERLAP_READY_SCOPE = 'native_warm_models_cached_text_coupled_vae_overlap_ready_v1'
NORMALIZATION_SHA = '23c901abf8b71ad565f1ae14d240908968da02d1dd3c1699e50e0b63365a02e3'
DECODE_AST_SHA = '423351c4d3ab73ad6913eea7e98423d36eb4e8f4d6044ba6be27535a485aaf3f'
MODELS = {
 'longlive': dict(model_id='longlive_1.3b', source_commit='2aa857f595d35c81526938d603ae7b1082922ac6',
  spec='experiments/erank3-longlive-policy-v1.artifacts.json',
  spec_sha256='cc30f6488dce2d9ed944a90c60270f61194a28c19a483168475cb32cec4479b1',
  archive_name='longlive_source', prefix='LongLive-2aa857f595d35c81526938d603ae7b1082922ac6/LongLive1.0',
  wrapper_sha256='57b7a636f5f333913a52ae2b7e64dcf023f21d7764ed702ea328af4b471e6026'),
 'rolling': dict(model_id='rolling_forcing_1.3b', source_commit='eb8705b2f00b7c361193b25aa3aae9e7edc94667',
  spec='experiments/erank3-rolling-native-v1.artifacts.json',
  spec_sha256='b1dbc91174fc0f4c3fe1ad0eb74c1103244538e9375fe301ff7625e4e6fe87bd',
  archive_name='rolling_source', prefix='Causal-Forcing-eb8705b2f00b7c361193b25aa3aae9e7edc94667/long_video',
  wrapper_sha256='bff68743b9e114f52d70e705a979255ac92da3410d69053a41cc2a21775d7ac4'),
}


def selected_model():
    model = os.environ.get(ENV)
    if model is not None and model not in MODELS:
        raise ValueError('native streaming model must be longlive or rolling')
    return model


def campaign_model(draft_model_id, *, overlap):
    """Bind an explicitly selected campaign to its native renderer."""
    model = selected_model()
    if not overlap:
        if model is not None:
            raise ValueError('the native renderer requires an explicit overlap campaign')
        return None
    if model is None or MODELS[model]['model_id'] != draft_model_id:
        raise ValueError('overlap campaign and native renderer model differ')
    if any(os.environ.get(name) != '4' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('overlap campaign requires the matched four-thread configuration')
    return validate_render_flags()


def completed_render_binding(proxy, history, prepared):
    """Bind the measured video to the receipt that joined the decoder and encoder."""
    from .erank_acceptance_pilot import bound_file

    receipt = proxy.last_evidence
    if (not receipt or receipt.get('decoder_and_encoder_joined') is not True
            or receipt.get('history_manifest_sha256') != history.manifest_sha256
            or receipt.get('history_head_sha256') != history.head_sha256
            or receipt.get('video', {}).get('sha256') != prepared.video_sha256):
        raise ValueError('completed render receipt does not match the measured video')
    path = proxy.run_dir / 'provenance' / 'accepted-stream' / (history.manifest_sha256 + '.json')
    return bound_file(proxy.run_dir, path)


def identity(model):
    row = MODELS[model]
    return dict(schema='coupled_native_vae_component_v1', native_model=model,
        native_model_id=row['model_id'], source_commit=row['source_commit'],
        artifact_spec_sha256=row['spec_sha256'], wrapper_sha256=row['wrapper_sha256'],
        native_vae_sha256=VAE_SHA, checkpoint_sha256=WEIGHTS_SHA,
        normalization_sha256=NORMALIZATION_SHA, native_decode_ast_sha256=DECODE_AST_SHA)


def validate_identity(value, model):
    if model not in MODELS or value != identity(model):
        raise ValueError('stream decoder does not match the coupled native model VAE')


def normalization(source, model):
    if hashlib.sha256(source).hexdigest() != MODELS[model]['wrapper_sha256']:
        raise ValueError('coupled native VAE wrapper bytes differ')
    tree = ast.parse(source.decode())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanVAEWrapper']
    if len(classes) != 1:raise ValueError('ambiguous native VAE wrapper')
    methods = {n.name:n for n in classes[0].body if isinstance(n, ast.FunctionDef)}
    decode = methods['decode_to_pixel']
    # Python 3.12 adds an empty type-parameter field to this Python 3.11 method.
    # Preserve any nonempty field so a change in the actual signature still fails.
    if getattr(decode, 'type_params', None) == []:
        del decode.type_params
    if hashlib.sha256(ast.dump(decode, include_attributes=False).encode()).hexdigest() != DECODE_AST_SHA:
        raise ValueError('native VAE decode math differs')
    values = {}
    for n in methods['__init__'].body:
        if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name in ('mean','std'):
                if name in values:raise ValueError('duplicate native normalization')
                values[name] = ast.literal_eval(n.value)
    digest = hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest != NORMALIZATION_SHA:raise ValueError('native VAE normalization differs')
    return values


def components(model, *, repository, model_root):
    from .rolling_forcing_assets import _contained_file
    from .rolling_forcing_native import _check_file
    if model not in MODELS:raise ValueError('unsupported coupled native VAE')
    row=MODELS[model]
    spec_path=_contained_file(repository,row['spec']);_check_file(spec_path,row['spec_sha256'])
    spec=json.loads(spec_path.read_text());items={x['name']:x for x in spec['artifacts']}
    archive_item=items[row['archive_name']];weights_item=items['wan_vae_checkpoint']
    if (archive_item['root']!='model' or weights_item['root']!='model'
            or weights_item['sha256']!=WEIGHTS_SHA):
        raise ValueError('native VAE artifact roots or weights differ')
    archive=_contained_file(model_root,archive_item['path']);_check_file(archive,archive_item['sha256'])
    weights=_contained_file(model_root,weights_item['path']);_check_file(weights,WEIGHTS_SHA)
    with tarfile.open(archive) as tar:
        def member(relative):
            entries=[entry for entry in tar.getmembers() if entry.name==row['prefix']+'/'+relative]
            if len(entries)!=1:
                raise ValueError('ambiguous coupled native VAE source member')
            entry=entries[0]
            if not entry.isfile() or not 0<entry.size<1024*1024:
                raise ValueError('native VAE source member is not a bounded regular file')
            return tar.extractfile(entry).read()
        source=member('wan/modules/vae.py');wrapper=member('utils/wan_wrapper.py')
    if hashlib.sha256(source).hexdigest()!=VAE_SHA:raise ValueError('native VAE implementation differs')
    return source,normalization(wrapper,model),weights,identity(model)


def validate_flags():
    from .video_encoding_policy import policy, SERIAL
    model=selected_model()
    if (model is None or os.environ.get('CFEXP_ACCEPTED_VIDEO_STREAM')!='1'
            or os.environ.get('CFEXP_RENDER_TARGET_PRIORITY')!='1' or policy()!=SERIAL):
        raise ValueError('native stream requires explicit model, priority and serial codec')
    for name in ('CFEXP_DRAFT_KV_REUSE','CFEXP_DRAFT_KV_FORK','CFEXP_CPU_HISTORY_BINDING',
                 'CFEXP_NATIVE_ROPE_COEFFICIENT_REUSE','CFEXP_NATIVE_FULL_LENGTH_ATTENTION',
                 'CFEXP_ACCEPTED_TARGET_PREFILL'):
        if os.environ.get(name,'0')!='0':
            raise ValueError('native stream cannot install Self-Forcing generation features')
    return model


def validate_render_flags():
    """Require the evaluator path that drains the accepted renderer."""
    from .video_only_decode import enabled

    model = validate_flags()
    if not enabled():
        raise ValueError('native rendering requires CFEXP_VIDEO_ONLY_DECODE=1')
    return model


def validate_worker(worker):
    """Only the original native backend owns its generation/cache semantics."""
    model=validate_flags()
    if worker._chunk_count not in (30,60,120):
        raise ValueError('native stream requires its declared duration horizon')
    if worker.role=='draft':
        if model=='longlive':
            from .longlive_policy_backend import LongLivePolicyBackend as Backend
        else:
            from .rolling_forcing_policy_backend import RollingPolicyBackend as Backend
        if (not isinstance(worker.backend,Backend)
                or worker._backend_model_state.get('model_id')!=MODELS[model]['model_id']
                or worker._backend_model_state.get('source_commit')!=MODELS[model]['source_commit']
                or worker._backend_model_state.get('vae_sha256')!=WEIGHTS_SHA):
            raise ValueError('native rendering worker has the wrong generation backend')
    elif worker.role=='target':
        from .krea_chunk_backend import KreaChunkBackend
        if (not isinstance(worker.backend,KreaChunkBackend)
                or worker._backend_model_state.get('model_id')!='krea-realtime-video-14b'):
            raise ValueError('native rendering requires the pinned Target worker')
    else:raise ValueError('invalid native renderer role')
