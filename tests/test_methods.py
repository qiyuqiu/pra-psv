import math
from types import SimpleNamespace
import pytest
import torch
from pra_psv.assets import checkpoint, pairs
from pra_psv.pra.inference import PRAMonitor
from pra_psv.psv.core import paired_targets
from pra_psv.psv.pairing import collect_pair
from pra_psv.psv.policy import decide
from pra_psv.psv.routing import PSVRouter
from pra_psv.integrations.agentdojo import visible_messages_for_call


def route(score, used=0, eligible=True, prediction=None):
    return decide(tool='read_document', observation_eligible=eligible,
                  pra_score=score, prediction=prediction, reviews_used=used,
                  read_only_tools=frozenset({'read_document'}),
                  margin_cutoff=.2140339875221252)


def test_margin_has_no_total_review_cap():
    assert route(.9, used=12).review
    assert not route(.1, used=12).review
    assert not route(.9, eligible=False).review


def test_uncertainty_and_tool_category_do_not_trigger():
    assert not route(.1, prediction=dict(selected=True, abstain=True,
                      unsupported_branch=False, out_of_support=False)).review
    assert not route(.1).review
    assert route(.1, prediction=dict(selected=True, abstain=False,
                 unsupported_branch=False, out_of_support=False)).review
    with pytest.raises(ValueError):
        route(math.nan)
    with pytest.raises(ValueError):
        route(.1, prediction=dict(selected=1, abstain=False,
              unsupported_branch=False, out_of_support=False))


def test_no_learned_inference_after_support_budget():
    calls=[]
    model=SimpleNamespace(predict=lambda p: calls.append(p))
    router=PSVRouter(model)
    decision,prediction=router.select_review(tool='write', observation_eligible=True,
                                           pra_score=.9,reviews_used=2,pack={})
    assert decision.review and prediction is None and calls == []


def test_pair_targets_mask_unknown_and_use_actual_outcomes():
    keep=dict(utility=True,attack_success=True)
    review=dict(utility=True,attack_success=False)
    assert paired_targets(keep,review) == [1.,1.,0.]
    assert paired_targets(dict(utility=None,attack_success=False),review) is None
    assert paired_targets(dict(keep,error='failure'),review) is None
    assert paired_targets(dict(keep,trace_ambiguity=True),review) is None


def test_pair_arms_are_independent_and_test_context_is_rejected():
    context=dict(id='x',group='g',split='train',state=[])
    seen=[]
    def rollout(c,force_review,teacher):
        seen.append(list(c['state']))
        c['state'].append(force_review)
        return dict(utility=True,attack_success=not force_review),{'causal':True}
    pair,_=collect_pair(context,rollout)
    assert seen == [[],[]] and context['state'] == []
    assert pair['targets'] == [1.,1.,0.]
    with pytest.raises(ValueError):
        collect_pair(dict(context,split='test'),rollout)


def test_later_same_batch_calls_are_invisible():
    calls=[SimpleNamespace(function='first'),SimpleNamespace(function='later')]
    messages=[dict(role='user',content='request'),dict(role='assistant',tool_calls=calls)]
    visible=visible_messages_for_call(messages,calls[:1],[])
    assert visible[-1]['tool_calls'] == calls[:1]
    assert messages[-1]['tool_calls'] == calls


def test_reference_weights_load_and_have_correct_shapes():
    monitor=PRAMonitor.from_release()
    out=monitor.score_features(torch.zeros(2,3584),[2,3])
    assert out.shape == (2,) and torch.isfinite(out).all()
    with pytest.raises(ValueError):
        monitor.score_features(torch.zeros(2,5),[2,3])
    assert checkpoint('psv_round1')['dim'] == 236


def test_cached_groups_do_not_overlap():
    for ri,n in [(0,241),(1,227)]:
        rows=pairs(ri)
        assert len(rows) == n
        train={r['group'] for r in rows if r['split']=='train'}
        select={r['group'] for r in rows if r['split']=='validation_select'}
        assert not train & select
        assert {r['split'] for r in rows} == {'train','validation_select'}
        for r in rows:
            assert set(r['pack']) <= {'views','numeric','pra_block','coverage','view_keys'}


def test_dh_timely_alarm_rule():
    from pra_psv.pra.dh.metrics import TaskPrediction,confusion_at
    harmful=TaskPrediction('h','QA',True,2,[1.,.1,.9])
    benign=TaskPrediction('b','QA',False,None,[1.,.1,.1])
    counts=confusion_at([harmful,benign],.5)
    assert counts.fn == 1 and counts.tn == 1 and counts.too_late == 1
