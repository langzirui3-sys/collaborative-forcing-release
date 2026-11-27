"""Fixed visual endpoints and descriptive forecasts; no threshold fitting.

Labels are reviewed before ranks. These helpers never infer visual labels from
rank, motion, color statistics, or the identity of the generating model.
"""
from __future__ import annotations
import math
from .motion_video_audit import require
from .prevention_rank_predictors import FEATURES

LABELS={'noncollapsed','severe_artifact','uncertain'}


def validate_labels(labels):
    require(isinstance(labels,list) and len(labels)==60 and all(x in LABELS for x in labels),
            'complete fixed 1Hz visual labels required')
    return labels


def onset_candidates(labels, *, possible=False):
    validate_labels(labels)
    allowed={'severe_artifact','uncertain'} if possible else {'severe_artifact'}
    return [i for i in range(58) if all(x in allowed for x in labels[i:i+3])]


def visual_endpoint(labels):
    validate_labels(labels)
    confirmed=onset_candidates(labels);possible=onset_candidates(labels,possible=True)
    onset=confirmed[0] if confirmed else None
    # Three confident healthy samples define a sustained observed recovery;
    # report subsequent confirmed three-sample failures separately.
    recovery=None;relapse=None
    if onset is not None:
        recovery=next((i for i in range(onset+3,58)
            if labels[i:i+3]==['noncollapsed']*3),None)
        if recovery is not None:
            relapse=next((i for i in confirmed if i>=recovery+3),None)
    return {'confirmed_onset_seconds':onset,'possible_onset_seconds':possible[0] if possible else None,
        'right_censored_no_confirmed_event':onset is None,'last_observed_sample_seconds':59,
        'last_fully_confirmable_onset_seconds':57,
        'uncertain_samples':labels.count('uncertain'),'noncollapsed_samples':labels.count('noncollapsed'),
        'severe_samples':labels.count('severe_artifact'),
        'normal_sample_fraction_bounds':[labels.count('noncollapsed')/60,
            (labels.count('noncollapsed')+labels.count('uncertain'))/60],
        'sustained_recovery_seconds':recovery,'confirmed_relapse_seconds':relapse}


def post_return_counts(labels, arm):
    validate_labels(labels)
    starts={'draft_only':0,'target_three_5s':7,'target_three_10s':12}
    require(arm in starts,'unknown intervention arm')
    start=starts[arm];future=labels[start:]
    return {'first_sample_seconds':start,'samples':len(future),
        'noncollapsed_samples':future.count('noncollapsed'),'uncertain_samples':future.count('uncertain'),
        'severe_samples':future.count('severe_artifact')}


def forecast_outcome(labels, seconds):
    """3s onset horizon, excluding uncertainty and unconfirmable end tails.

    The last assessable onset is 57s (confirmation uses samples 57,58,59).
    Hence a complete 3s forecasting window ends at <=57s, not at video end.
    At risk requires no earlier possible three-sample collapse. Uncertainty in
    any future onset/confirmation sample excludes the forecast, never a negative.
    """
    validate_labels(labels)
    require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds>=0,
            'invalid forecast time')
    if seconds+3>57:return None
    possible=onset_candidates(labels,possible=True)
    if possible and possible[0]<=seconds:return None
    future=range(math.floor(seconds)+1,math.floor(seconds+3)+3)
    if any(labels[i]=='uncertain' for i in future):return None
    confirmed=onset_candidates(labels)
    return int(bool(confirmed and seconds<confirmed[0]<=seconds+3))


def roc_auc(scores, outcomes):
    require(len(scores)==len(outcomes) and all(type(y) is int and y in (0,1) for y in outcomes),
            'invalid binary forecast population')
    require(all(type(s) in (int,float) and math.isfinite(s) for s in scores),'invalid risk scores')
    positives=[s for s,y in zip(scores,outcomes) if y==1]
    negatives=[s for s,y in zip(scores,outcomes) if y==0]
    if not positives or not negatives:return None
    return sum((p>n)+.5*(p==n) for p in positives for n in negatives)/(len(positives)*len(negatives))


def predictor_summary(feature_rows,labels):
    require(len(feature_rows)==120 and [r['chunk'] for r in feature_rows]==list(range(120)),
            'complete ordered feature rows required')
    eligible=[(r,forecast_outcome(labels,r['video_seconds'])) for r in feature_rows]
    def summarize(feature, common):
        rows=[(r[feature],y) for r,y in eligible if y is not None and r[feature] is not None
            and (not common or all(r[k] is not None for k in FEATURES))]
        scores=[s for s,y in rows];outcomes=[y for s,y in rows]
        return {'eligible_timepoints':len(rows),'positive_timepoints':sum(outcomes),
            'negative_timepoints':len(rows)-sum(outcomes),'roc_auc':roc_auc(scores,outcomes),
            'risk_direction':'larger is higher risk, fixed prospectively'}
    return {'visual':visual_endpoint(labels),
        'at_risk_complete_future_timepoints':sum(y is not None for r,y in eligible),
        'common':{k:summarize(k,True) for k in FEATURES},
        'own_coverage':{k:summarize(k,False) for k in FEATURES}}
