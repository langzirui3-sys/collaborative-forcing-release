"""Prospectively fixed, causal eRank features. No fitted threshold or quality gate."""
from __future__ import annotations
import math
from .motion_video_audit import require
from .single_target_recovery import CHUNKS, video_time

FEATURES = ('step_fall', 'step_rise', 'step_magnitude', 'sustained_fall_5s', 'drawdown_10s')


def _window(values, index, seconds):
    end = video_time(index)
    start = end - seconds
    if start < 0:
        return None
    ids = [i for i in range(index+1) if video_time(i) >= start-1e-9]
    if not ids or video_time(ids[0]) > start+1e-9 or any(values[i] is None for i in ids):
        return None
    # Only contiguous Draft observations; Target gaps are never interpolated.
    if any(video_time(j)-video_time(i) > .5+1e-9 for i,j in zip(ids,ids[1:])):
        return None
    return [(video_time(i), values[i]) for i in ids]


def causal_features(values):
    require(len(values) == CHUNKS, 'incomplete rank trajectory')
    require(all(x is None or (type(x) in (int,float) and math.isfinite(x) and 1 <= x <= 256)
                for x in values), 'invalid fixed-probe rank')
    rows=[]
    for i,x in enumerate(values):
        row={'chunk':i,'video_seconds':video_time(i),**{key:None for key in FEATURES}}
        if x is not None:
            if i > 0 and values[i-1] is not None:
                delta=x-values[i-1]
                row.update(step_fall=-delta,step_rise=delta,step_magnitude=abs(delta))
            short=_window(values,i,5)
            if short is not None:
                mt=sum(t for t,v in short)/len(short)
                mv=sum(v for t,v in short)/len(short)
                slope=sum((t-mt)*(v-mv) for t,v in short)/sum((t-mt)**2 for t,v in short)
                row['sustained_fall_5s']=-slope
            long=_window(values,i,10)
            if long is not None:
                peak=max(v for t,v in long)
                row['drawdown_10s']=(peak-x)/peak
        rows.append(row)
    return rows


def features_for_trace(trace):
    from .intervention_erank_analysis import observations
    return causal_features(observations(trace))
