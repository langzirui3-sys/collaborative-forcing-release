"""Opt-in one-shot native CUDA gate with process-owned measured observations."""
import os
import time

from .joint_gate_profile import ENV,forward_observations,append
from .longlive_joint_worker import JointWorker,ACTION
from .longlive_joint_handoff import require


class TrackedJointWorker(JointWorker):
    def __init__(self,**kwargs):
        require(os.environ.get(ENV)=='1','tracked joint worker requires explicit diagnostic mode')
        super().__init__(**kwargs)
        from .stage_timing import _unsealed_directory
        self.tracking_root=_unsealed_directory(self.run_dir)
        self.t=self.backend.torch
        require(self.t.cuda.is_available(),'joint telemetry requires actual CUDA')
        self.t.cuda.reset_peak_memory_stats()
        directory=self.tracking_root/'provenance'/'tracking';directory.mkdir(parents=True,exist_ok=True)
        self.tracking_path=directory/f'{self.role}-requests-{os.getpid()}.jsonl';self.tracking_path.open('x').close()
        self.tracking_sequence=0

    def handle(self,request):
        before=time.perf_counter()
        if request.get('action')==ACTION:
            from .stage_timing import timing_session,stage
            with timing_session(self.run_dir,'worker-draft-joint',dict(action=ACTION,role=self.role,
                    request_id=request.get('request_id'),chunk_index=None)) as session:
                require(session is not None,'tracked joint gate requires stage timing')
                session.attach_cuda(self.t,self.backend.device)
                with forward_observations(self.backend,self.run_dir),stage('complete_joint_oracle',cuda=True):
                    value=super().handle(request)
        else:value=super().handle(request)
        append(self.tracking_path,dict(schema='longlive_joint_worker_observation_v1',role=self.role,pid=os.getpid(),
            rank=0,sequence=self.tracking_sequence,action=request.get('action'),request_id=request.get('request_id'),
            request_wall_seconds=time.perf_counter()-before,peak_allocated_bytes=self.t.cuda.max_memory_allocated(),
            peak_reserved_bytes=self.t.cuda.max_memory_reserved(),
            scope='whole diagnostic RPC including setup/host/parity checks; not production latency'))
        self.tracking_sequence+=1;return value


def main(argv=None):
    from . import correction_worker as base
    original=base.CorrectionWorker;base.CorrectionWorker=TrackedJointWorker
    try:return base.main(argv)
    finally:base.CorrectionWorker=original


if __name__=='__main__':raise SystemExit(main())
