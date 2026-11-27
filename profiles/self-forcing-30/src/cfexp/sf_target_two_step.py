"""Opt-in position-dependent Krea corrections for Self-Forcing transfer."""

import os

from .krea_chunk_backend import (
    AdapterError, MAIN_TABLE_FOUR_STEP_SCHEDULE, copy_explicit_session_noise,
    validate_native_correction_state, validate_target_forward_trace,
)
from .self_forcing_frame_adapter import (
    attach_adapter_timing, sampled_kv_cache_witness, timed_forward_stages,
    witness_index_execution_payload,
)
from .stage_timing import stage


ENV = "CFEXP_SF_TARGET_TWO_STEP"
SCHEDULE = (1000.0, 500.0)
LOW_NOISE_REFINEMENT = (1000.0, 250.0)
LOW_NOISE_ENV = "CFEXP_SF_TARGET_LOW_NOISE_REFINEMENT"
FOLLOWUP_ENV = "CFEXP_SF_TARGET_BURST_INTERIOR_ONE_STEP"
BURST_ENTRY_ENV = "CFEXP_SF_TARGET_BURST_ENTRY_FOUR_STEP"
BURST_SECOND_ENV = "CFEXP_SF_TARGET_BURST_SECOND_TWO_STEP"
BURST_EXIT_ENV = "CFEXP_SF_TARGET_BURST_EXIT_TWO_STEP"
BURST_EXTRA_ENV = "CFEXP_SF_TARGET_BURST_EXTRA_ONE_STEP"
EARLY_NATIVE_ENV = "CFEXP_SF_TARGET_EARLY_NATIVE_BURSTS"
LONG_OPENING_ENV = "CFEXP_SF_TARGET_LONG_OPENING"
FOLLOWUP_SCHEDULE = (1000.0,)
ARM = "rank_burst4_split_p2_k36"
NATIVE_FACTORY = "cfexp.correction_runtime:build_target_backend"


def enabled(env=os.environ):
    value = env.get(ENV, "0")
    if value not in ("0", "1"):
        raise AdapterError(f"{ENV} must be 0 or 1")
    return value == "1"


def one_step_followups(env=os.environ):
    value = env.get(FOLLOWUP_ENV, "0")
    if value not in ("0", "1") or (value == "1" and not enabled(env)):
        raise AdapterError("One-step Target continuations require the two-step Target worker")
    return value == "1"


def early_native_burst_count(env=os.environ):
    value = env.get(EARLY_NATIVE_ENV, "0")
    if value not in ("0", "2", "3") or (value != "0" and not one_step_followups(env)):
        raise AdapterError("The early native preset requires one-step continuation")
    return int(value)


def long_opening(env=os.environ):
    value = env.get(LONG_OPENING_ENV, "0")
    if value not in ("0", "1") or (value == "1" and not early_native_burst_count(env)):
        raise AdapterError("Long Target opening requires two or three native bursts")
    return value == "1"


def schedules_for_burst(later_schedules, *, burst_index, early_native_bursts):
    if burst_index < early_native_bursts:
        return (MAIN_TABLE_FOUR_STEP_SCHEDULE,) * len(later_schedules)
    return later_schedules


def burst_schedules(env=os.environ):
    followups = one_step_followups(env)
    entry_value = env.get(BURST_ENTRY_ENV, "0")
    second_value = env.get(BURST_SECOND_ENV, "0")
    exit_value = env.get(BURST_EXIT_ENV, "0")
    extra_value = env.get(BURST_EXTRA_ENV, "0")
    overrides = (entry_value, second_value, exit_value, extra_value)
    low_noise = env.get(LOW_NOISE_ENV, "0")
    if low_noise not in ("0", "1") or (low_noise == "1" and not followups):
        raise AdapterError("Low-noise refinement requires a mixed-step Target burst")
    if any(value not in ("0", "1") for value in overrides) or (not followups and "1" in overrides):
        raise AdapterError("Target slot overrides require the one-step burst continuation mode")
    early = early_native_burst_count(env)
    width = 8 if long_opening(env) else 4
    if early == 3:
        if "1" in overrides or low_noise == "1":
            raise AdapterError("The early native preset already specifies both schedule phases")
        return (FOLLOWUP_SCHEDULE,) * width
    if early == 2:
        entry_only = width == 4 and env.get(BURST_EXIT_ENV) == "0"
        if overrides != ("0", "0", "0" if entry_only else "1", "0"):
            raise AdapterError("Two early native bursts require two-step entry and exit afterward")
    if not followups:
        return None
    refinement = LOW_NOISE_REFINEMENT if low_noise == "1" else SCHEDULE
    entry_schedule = MAIN_TABLE_FOUR_STEP_SCHEDULE if entry_value == "1" else refinement
    second_schedule = refinement if second_value == "1" else FOLLOWUP_SCHEDULE
    exit_schedule = refinement if exit_value == "1" else FOLLOWUP_SCHEDULE
    interior = (FOLLOWUP_SCHEDULE,) * (width - 3 + int(extra_value == "1"))
    schedules = (entry_schedule, second_schedule, *interior, exit_schedule)
    if low_noise == "1" and refinement not in schedules:
        raise AdapterError("Low-noise refinement requires a two-step slot")
    return schedules


def schedule_metadata(env=os.environ):
    schedules = burst_schedules(env)
    if schedules is not None:
        result = {"target_burst_denoising_schedules": [list(schedule) for schedule in schedules]}
        early = early_native_burst_count(env)
        if early:
            result["target_early_native_bursts"] = early
        return result
    if enabled(env):
        return {"target_nonroot_denoising_schedule": list(SCHEDULE)}
    return {}


def validate_correction_trace(trace, schedule, *, chunk_index, expected_nonroot_schedule):
    expected = MAIN_TABLE_FOUR_STEP_SCHEDULE if chunk_index == 0 else expected_nonroot_schedule
    if expected == MAIN_TABLE_FOUR_STEP_SCHEDULE:
        return validate_target_forward_trace(
            trace, schedule, history_frames=3 * chunk_index, expected_schedule=expected,
        )
    if expected not in (FOLLOWUP_SCHEDULE, SCHEDULE, LOW_NOISE_REFINEMENT) or tuple(schedule) != expected:
        raise AdapterError("Target denoising schedule differs from the required schedule")
    if len(trace) != 1 + len(expected):
        raise AdapterError("Target total forwards differ from its denoising steps plus context")
    if tuple(item.get("timestep") for item in trace) != (0.0, *expected):
        raise AdapterError("Target forward timesteps differ from the required trace")
    if any(item.get("current_start") != 4680 or item.get("batch_timesteps_equal") is not True
           for item in trace):
        raise AdapterError("Target forward used an invalid KV cursor or batch timestep")
    return list(trace[1:])


def process_factory(argv, **kwargs):
    """Select the alternate Target factory in this streaming trial only."""
    from .accepted_stream_worker import process_factory as start_stream_worker

    if not enabled():
        raise AdapterError("Two-step Target worker requires explicit opt-in")
    command = list(argv)
    index = command.index("--backend-factory") + 1
    if command[index] == NATIVE_FACTORY:
        command[index] = "cfexp.sf_target_two_step:build_target_backend"
    return start_stream_worker(command, **kwargs)


def build_target_backend(config):
    from .correction_runtime import build_target_backend as build_native

    chunk_count = 120 if long_opening() else 60
    if not enabled() or config.get("chunk_count") != chunk_count:
        raise AdapterError("Reduced-step Target worker requires its frozen chunk count")
    schedules = burst_schedules()
    return TwoStepTarget(build_native(config), schedules=schedules,
                         early_native_bursts=early_native_burst_count())


class TwoStepTarget:
    def __init__(self, native, *, schedules, early_native_bursts=0):
        self.native = native
        self.schedules = schedules
        self.early_native_bursts = early_native_bursts
        self.completed_bursts = 0
        self.last_correction = None
        self.burst_offset = 0

    def _next_burst_offset(self, chunk_index):
        previous = self.last_correction
        if previous is None:
            return 0
        last_offset = len(self.schedules) - 1
        if chunk_index == previous + 1 and self.burst_offset < last_offset:
            return self.burst_offset + 1
        if chunk_index > previous + 1 and self.burst_offset == last_offset:
            return 0
        raise AdapterError("Target chunks must complete each declared consecutive burst")

    def __getattr__(self, name):
        return getattr(self.native, name)

    def generate_chunk(self, *, prompt, history_latents, initial_noise, internal_seed, chunk_index):
        native = self.native
        if chunk_index == 0:
            if (self.schedules is not None and self.last_correction is not None
                    and self.burst_offset != len(self.schedules) - 1):
                raise AdapterError("The previous Target trajectory ended inside a burst")
            self.last_correction = None
            self.burst_offset = 0
            self.completed_bursts = 0
            return native.generate_chunk(
                prompt=prompt, history_latents=history_latents, initial_noise=initial_noise,
                internal_seed=internal_seed, chunk_index=chunk_index,
            )
        torch = native.torch
        session = getattr(native, "_correction_session", None)
        if session is None or native._correction_prompt != prompt or not 0 < chunk_index < native.chunk_count:
            raise AdapterError("Two-step Target requires the same prompt and its native root session")
        if native._persistent_session is not None:
            raise AdapterError("Two-step correction cannot overlap a persistent Target trajectory")
        history_frames = 3 * chunk_index
        if (tuple(history_latents.shape) != (1, history_frames, 16, 60, 104)
                or tuple(initial_noise.shape) != (1, 3, 16, 60, 104)):
            raise AdapterError("Two-step Target received an invalid history or noise shape")
        attach_adapter_timing(torch, native.device)
        with stage("history_bind", detail="target", cuda=True), torch.inference_mode():
            history = history_latents.to(device=native.device, dtype=torch.bfloat16).contiguous()
            if not bool(torch.isfinite(history.float()).all().item()):
                raise AdapterError("Two-step Target accepted history contains non-finite values")
            destination = session.all_latents[:, :history_frames]
            destination.copy_(history)
            if not bool(torch.equal(destination, history)):
                raise AdapterError("Two-step Target session did not preserve the accepted history")
            session.block_idx = chunk_index
            session.current_start_frame = history_frames
            session.resume_latents = None
        offset = self._next_burst_offset(chunk_index) if self.schedules is not None else 0
        selected_schedule = SCHEDULE
        if self.schedules is not None:
            selected_schedule = schedules_for_burst(
                self.schedules, burst_index=self.completed_bursts,
                early_native_bursts=self.early_native_bursts,
            )[offset]
        session.denoising_step_list = list(selected_schedule)
        with stage("transfer", detail="target_noise_session_copy", cuda=True):
            noise = initial_noise.to(device=native.device, dtype=torch.bfloat16).contiguous()
            copy_explicit_session_noise(
                session=session, noise=noise, history_frames=history_frames, torch_module=torch,
            )
        session.rnd.manual_seed(internal_seed)
        with stage("conditioning", detail="target", cuda=True):
            condition = native._condition(prompt)
        session.conditional_dict = condition
        session.current_prompt_embeds = condition["prompt_embeds"]

        trace = []
        witness = None

        def capture(_module, _args, kwargs):
            nonlocal witness
            values = kwargs["timestep"].detach().reshape(-1).to("cpu").tolist()
            trace.append({
                "timestep": float(values[0]),
                "batch_timesteps_equal": len(set(values)) == 1,
                "current_start": kwargs["current_start"],
            })
            if values[0] > 0 and witness is None:
                with stage("kv_witness", detail="target", cuda=True):
                    witness = sampled_kv_cache_witness(
                        native.models.pipeline.kv_cache1,
                        getattr(native.models.pipeline, "crossattn_cache", None),
                    )

        transformer = native.models.transformer
        hook = transformer.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with timed_forward_stages(transformer, prefill_count=1), torch.inference_mode():
                session.generate_block(native.models)
        finally:
            hook.remove()
        schedule = session.denoising_step_list
        denoising = validate_correction_trace(
            trace, schedule, chunk_index=chunk_index, expected_nonroot_schedule=selected_schedule,
        )
        if witness is None:
            raise AdapterError("Two-step Target omitted its pre-denoising KV witness")
        with stage("transfer", detail="target_result_d2h", cuda=True):
            candidate = session.last_pred.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        if tuple(candidate.shape) != (1, 3, 16, 60, 104) or not bool(torch.isfinite(candidate.float()).all().item()):
            raise AdapterError("Two-step Target generated an invalid clean latent")
        state = {
            "schema": "krea_native_correction_state_v1",
            "native_session_id": native._correction_session_id,
            "session_recreated": False,
            "chunk_index": chunk_index,
            "accepted_history_frames": history_frames,
            "bounded_context_rebase": True,
            "context_frames": 3,
        }
        validate_native_correction_state(state, chunk_index=chunk_index)
        self.last_correction = chunk_index
        self.burst_offset = offset
        if self.schedules is not None and offset == len(self.schedules) - 1:
            self.completed_bursts += 1
        return {
            "latent": candidate, "native_correction_state": state,
            "kv_witness_sha256": witness[0], "kv_witness": witness[1],
            **witness_index_execution_payload(witness),
            "forward_trace": trace, "denoising_forward_trace": denoising,
            "denoising_schedule": list(schedule), "explicit_initial_noise_verified": True,
        }
