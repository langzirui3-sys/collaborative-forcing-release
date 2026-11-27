"""One-shot diagnostic worker; never shared with a timed generation path."""
import json
import os
import subprocess

from .accepted_stream_worker import StreamingCorrectionWorker
from .correction_worker import _exact_keys
from .longlive_joint_handoff import ENV, require
from .longlive_joint_oracle import run, CASES, SCHEDULE
from .prefix_worker import file_hash

ACTION = 'longlive_joint_gate'


class JointWorker(StreamingCorrectionWorker):
    def __init__(self, **kwargs):
        require(os.environ.get(ENV) == '1', 'joint worker requires explicit diagnostic opt-in')
        super().__init__(**kwargs)
        if self.role == 'draft':
            from .longlive_policy_worker import OwnedLongLivePolicyBackend
            require(type(self.backend) is OwnedLongLivePolicyBackend, 'joint worker requires the native LongLive backend')
        self.joint_finished = False

    def _validate_common(self, request):
        if request.get('action') == ACTION:
            identity, _ = super()._validate_common(dict(request, action='model_state'))
            require(not self.poisoned and not self.joint_finished, 'joint diagnostic worker already consumed')
            return identity, ACTION
        return super()._validate_common(request)

    def handle(self, request):
        identity, action = self._validate_common(request)
        if action != ACTION:
            require(not (self.joint_finished and action not in {'shutdown', 'model_state'}),
                    'one-shot joint worker cannot resume ordinary operations')
            return super().handle(request)
        _exact_keys(request, self._COMMON_KEYS | {'prompt', 'history_latent_paths', 'noise_paths', 'output_paths'},
                    label='native joint diagnostic')
        require(self.role == 'draft' and isinstance(request['prompt'], str) and request['prompt'].strip(),
                'joint gate requires a Draft prompt')
        history, paths = self._history(request['history_latent_paths'], expected_count=12)
        raw = request['noise_paths']
        require(isinstance(raw, list) and len(raw) == 28, 'joint gate requires every explicit re-noising draw')
        loaded = [self._load_input_tensor(x, label='joint noise', key='noise') for x in raw]
        paths += [x[0] for x in loaded]
        require(len(set(paths)) == len(paths), 'joint inputs alias')
        output, target = self._output_map(request['output_paths'], required={'report'})['report']
        require(output not in paths, 'joint report aliases an input')
        report = dict(schema='longlive_joint_cuda_gate_v1', passed=False, prompt=request['prompt'],
            cases=[list(x) for x in CASES], native_denoising_schedule=list(SCHEDULE),
            native_local_frames=12, native_sink_frames=3, clean_update_frames=3,
            formal_quality_result=False, no_end_to_end_speed_claim=True,
            input_files=[dict(path=p, sha256=file_hash(self, p)) for p in paths], rows=[])
        try:
            with self._render_gpu_lock:
                run(self.backend, prompt=request['prompt'], history=history.to(self.backend.device),
                    noises=[x.to(self.backend.device) for _, x in loaded], report=report)
                self.joint_finished = True
            return identity, action, report
        except BaseException as error:
            report.update(passed=False, error=str(error))
            raise self._poison(error) from error
        finally:
            with target.open('x') as out:
                json.dump(report, out, indent=2, allow_nan=False)


def process_factory(argv, **kwargs):
    require(os.environ.get(ENV) == '1' and list(argv[1:3]) == ['-m', 'cfexp.correction_worker'],
            'joint launcher requires its explicit diagnostic entrypoint')
    command = list(argv); command[2] = 'cfexp.longlive_joint_worker'
    return subprocess.Popen(command, **kwargs)


def main(argv=None):
    from . import correction_worker as base
    require(os.environ.get(ENV) == '1', 'joint worker requires explicit diagnostic opt-in')
    original = base.CorrectionWorker; base.CorrectionWorker = JointWorker
    try:
        return base.main(argv)
    finally:
        base.CorrectionWorker = original


if __name__ == '__main__':
    raise SystemExit(main())
