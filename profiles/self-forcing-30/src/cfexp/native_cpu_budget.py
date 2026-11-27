"""Audit actual worker CPU thread budgets without changing native model math."""
import json
from .erank_acceptance_pilot import bound_file


def audit(run_dir, *, chunk_count, arm):
    expected = {i: ('target' if arm == 'target' or i in (0,8,9,10,11) else 'draft')
                for i in range(chunk_count)}
    seen = {}; files = []
    for path in sorted((run_dir/'provenance/stage-timing').glob('worker-*.jsonl')):
        files.append(bound_file(run_dir, path))
        for line in path.read_text().splitlines():
            row = json.loads(line); meta = row['metadata']
            if ('torch_num_threads' in meta and
                    (type(meta['torch_num_threads']) is not int or meta['torch_num_threads'] != 4
                     or meta.get('cpu_thread_environment') != {'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})):
                raise ValueError('timed native worker did not apply its CPU thread budget')
            if meta.get('action') not in ('generate', 'generate_persistent'):
                continue
            index = meta.get('chunk_index'); role = meta.get('role')
            if (type(index) is not int or role not in ('draft','target')
                    or index in seen or expected.get(index) != role
                    or row.get('source') != 'worker-'+role or row.get('status') != 'completed'
                    or type(meta.get('torch_num_threads')) is not int or meta['torch_num_threads'] != 4
                    or meta.get('cpu_thread_environment') != {'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}):
                raise ValueError('CPU thread budget execution evidence differs')
            seen[index] = role
    if seen != expected:
        raise ValueError('CPU thread budget does not cover the full accepted trajectory')
    return {'schema':'native_cpu_thread_budget_execution_v1', 'intraop_threads':4,
            'covered_chunks':chunk_count,'worker_timing_files':files}
