"""Diagnostic module comparisons; hooks observe but never replace outputs.

References are the last clean B3 call of an independent native prefix.
The joint call compares the corresponding clean group, not its noisy suffix.
CPU copies deliberately make this unsuitable for production timing.
"""
from contextlib import contextmanager
from functools import partial

from .longlive_joint_handoff import require


PATHS = tuple('blocks.0.' + path for path in (
    'norm1', 'self_attn.q', 'self_attn.k', 'self_attn.v',
    'self_attn.norm_q', 'self_attn.norm_k', 'self_attn.o', 'self_attn',
    'norm3', 'cross_attn.q', 'cross_attn.o', 'cross_attn',
    'norm2', 'ffn.0', 'ffn.1', 'ffn.2', 'ffn',
)) + ('blocks.0', 'blocks.1.norm1', 'blocks.1.self_attn.q',
     'blocks.1.self_attn.k', 'blocks.1.self_attn.v')


def difference(actual, expected, *, torch):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype,
            'module diagnostic tensor geometry or dtype changed')
    a, b = actual.detach().float(), expected.detach().float()
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
            'module diagnostic encountered non-finite values')
    delta = a - b
    numerator, denominator = float(torch.linalg.vector_norm(delta)), float(torch.linalg.vector_norm(b))
    return dict(exact=bool(torch.equal(actual, expected)), shape=list(actual.shape),
        dtype=str(actual.dtype), different_elements=int(torch.count_nonzero(actual != expected)),
        maximum_absolute_difference=float(delta.abs().max()),
        l2_difference=numerator, reference_l2=denominator,
        relative_l2_difference=(numerator / denominator if denominator else None))


class ModuleTrace:
    def __init__(self, model, *, chunk_tokens, torch, report):
        require(type(chunk_tokens) is int and chunk_tokens > 0, 'invalid native B3 token count')
        self.model, self.chunk_tokens, self.torch, self.report = model, chunk_tokens, torch, report
        self.mode = None
        self.groups = None
        self.references = {}
        self.handles = []
        report.update(schema='longlive_joint_module_trace_v1',
            reference='independent_native_prefix_last_clean_B3',
            compared_group='last_clean_B3_before_noisy_suffix',
            no_timing_claim=True, comparisons=[], first_difference=None)

    def _observe(self, label, tensor):
        if self.mode is None:
            return
        require(isinstance(tensor, self.torch.Tensor) and tensor.ndim == 3
                and tensor.shape[0] == 1, 'module trace requires one token sequence: ' + label)
        if self.mode == 'native':
            require(tensor.shape[1] == self.chunk_tokens and label not in self.references,
                    'native module trace repeated or omitted a B3 shape: ' + label)
            self.references[label] = tensor.detach().to(device='cpu', copy=True)
            return
        require(tensor.shape[1] == self.groups * self.chunk_tokens
                and label in self.references, 'joint module trace geometry/reference differs: ' + label)
        require(not any(x['module'] == label for x in self.report['comparisons']),
                'joint module trace repeated a module: ' + label)
        selected = tensor[:, -2*self.chunk_tokens:-self.chunk_tokens]
        proof = difference(selected.detach().to(device='cpu', copy=True),
                           self.references[label], torch=self.torch)
        self.report['comparisons'].append(dict(module=label, **proof))
        if not proof['exact'] and self.report['first_difference'] is None:
            self.report['first_difference'] = label

    def _before(self, path, _module, args):
        if self.mode is not None:
            require(bool(args), 'module trace missing positional video input: ' + path)
            self._observe(path + ':input', args[0])

    def _after(self, path, _module, _args, output):
        if self.mode is not None:
            self._observe(path + ':output', output[0] if isinstance(output, tuple) else output)

    def __enter__(self):
        try:
            for path in PATHS:
                module = self.model.get_submodule(path)
                self.handles.append(module.register_forward_pre_hook(partial(self._before, path)))
                self.handles.append(module.register_forward_hook(partial(self._after, path)))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.references.clear()
        self.mode = None

    @contextmanager
    def native(self):
        require(self.mode is None and not self.references, 'module trace native capture already consumed')
        self.mode = 'native'
        try:
            yield
        finally:
            self.mode = None

    @contextmanager
    def joint(self, external):
        require(self.mode is None and len(self.references) == 2*len(PATHS)
                and type(external) is int and 1 <= external <= 4,
                'module trace requires complete native references and one to four clean blocks')
        self.mode, self.groups = 'joint', external + 1
        try:
            yield
        finally:
            self.mode = self.groups = None


def validate_trace(report):
    labels = {path + suffix for path in PATHS for suffix in (':input', ':output')}
    rows = report.get('comparisons', [])
    require(report.get('schema') == 'longlive_joint_module_trace_v1'
            and report.get('reference') == 'independent_native_prefix_last_clean_B3'
            and report.get('compared_group') == 'last_clean_B3_before_noisy_suffix'
            and report.get('no_timing_claim') is True
            and report.get('first_difference') is None
            and len(rows) == len(labels) and {x.get('module') for x in rows} == labels
            and all(x.get('exact') is True and x.get('different_elements') == 0
                    and x.get('maximum_absolute_difference') == 0
                    and x.get('l2_difference') == 0 for x in rows),
            'joint module trace is incomplete or not native-exact')
