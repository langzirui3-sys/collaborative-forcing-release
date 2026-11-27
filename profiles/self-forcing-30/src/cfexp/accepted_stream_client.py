"""Bind an accepted renderer to the evaluator's normal immutable video evidence."""
import os

from .accepted_stream import ENV, PRIORITY_ENV, STREAM_CHUNK_COUNTS


def enabled():
    value = os.environ.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError('invalid accepted stream switch')
    return value == '1'


class StreamDecoderProxy:
    def __init__(self, draft, renderer, run_dir, *, chunk_count=30):
        if type(chunk_count) is not int or chunk_count not in STREAM_CHUNK_COUNTS:
            raise ValueError("unsupported accepted stream duration")
        self.chunk_count = chunk_count
        self.draft, self.renderer, self.run_dir = draft, renderer, run_dir
        self.history = None
        self.last_evidence = None
        flag=os.environ.get(PRIORITY_ENV,'0')
        if flag not in ('0','1'):
            raise ValueError('invalid render priority flag')
        self.priority=flag=='1'
        self.render_allowed=True
        from .native_vae_component import selected_model, validate_identity
        model = selected_model()
        preparation = {}
        if model is not None:
            from pathlib import Path
            from .metadata import find_git_root
            preparation['native_repository_root'] = str(find_git_root(Path.cwd()))
        self.decoder = renderer._request('stream_prepare', preparation)
        from .streaming_vae_probe import VAE_SHA, WEIGHTS_SHA
        if (self.decoder.get('native_vae_sha256') != VAE_SHA
                or self.decoder.get('checkpoint_sha256') != WEIGHTS_SHA
                or self.decoder.get('separate_causal_decoder_state') is not True):
            raise ValueError('stream decoder is not the frozen separate native VAE')
        if model is not None:
            validate_identity(self.decoder.get('coupled_native_identity'), model)

    def __getattr__(self, name):
        return getattr(self.draft, name)

    def append(self, history):
        if self.history is not None and history.prefix_before(self.history.next_chunk_index) != self.history:
            raise ValueError('renderer callback changed the accepted prefix')
        payload = self.renderer._request('stream_append', {'history': history.to_dict()})
        if payload != {'history_head_sha256': history.head_sha256,
                       'latent_chunk_count': history.next_chunk_index}:
            raise ValueError('renderer acknowledged another accepted prefix')
        self.history = history

    def before_generate(self, role):
        if not self.priority:
            return
        if role not in ('draft','target'):
            raise ValueError('invalid next generation role')
        allowed=role!=self.renderer.role
        if allowed==self.render_allowed:
            return
        result=self.renderer._request('stream_set_role',{'generation_role':role})
        if result!={'allowed':allowed,'generation_role':role,'renderer_worker_role':self.renderer.role}:
            raise ValueError('renderer acknowledged another generation role')
        self.render_allowed=allowed

    def decode_video(self, *, history, output_video_path, fps=24):
        from .correction_runtime import DecodeResult, _write_immutable_json
        from .correction_worker import _input_path, _output_path, _sha256_file
        from .stage_timing import _unsealed_directory
        from .video_only_decode import validate_pixels
        _unsealed_directory(self.run_dir)
        if self.history != history or fps != 24 or history.next_chunk_index != self.chunk_count:
            raise ValueError('requested video differs from the accepted stream')
        result = self.renderer._request('stream_finish', {'history': history.to_dict()})
        from .native_vae_component import selected_model, validate_identity
        model = selected_model()
        if model is not None:
            validate_identity(result.get('decoder', {}).get('coupled_native_identity'), model)
            if result['decoder'] != self.decoder:
                raise ValueError('native stream decoder identity changed during generation')
        if (result['history_manifest_sha256'] != history.manifest_sha256
                or result['history_head_sha256'] != history.head_sha256
                or result['latent_chunk_count'] != self.chunk_count
                or result['decoder_and_encoder_joined'] is not True
                or [x['segment_sha256'] for x in result['chunks']] != [x.segment_sha256 for x in history.segments]):
            raise ValueError('completed stream is not bound to the complete accepted history')
        frame_count = self.chunk_count * 12 - 3
        validate_pixels(result['pixels'], frame_count)
        if result['video']['frames'] != frame_count or result['video']['fps'] != fps:
            raise ValueError('accepted video differs from requested geometry')
        video = {k: result['video'][k] for k in ('path','sha256','nbytes','fps','frames')}
        _, source = _input_path(self.run_dir, video['path'], label='accepted stream video')
        if _sha256_file(source) != (video['sha256'], video['nbytes']):
            raise ValueError('accepted stream video changed after encoder completion')
        relative, path = _output_path(self.run_dir, output_video_path, label='stream video alias')
        os.link(source, path)
        video['path'] = relative
        self.last_evidence = result
        _write_immutable_json(self.run_dir/'provenance'/'accepted-stream'/
                              (history.manifest_sha256+'.json'), result)
        self.history = None
        self.render_allowed=True
        return DecodeResult(frames=result['pixels'], video=video)
