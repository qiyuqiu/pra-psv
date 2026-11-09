"""Action-aware causal views. No task/attack IDs, outcomes, oracle plans or future text."""
import json, math, re
from .core import content, digest, features, can_review

EXTRA_NAMES = ('review_eligible','budget_used','budget_remaining','turns_remaining',
 'last_failed','consecutive_failures','consecutive_same_action','same_action_seen',
 'recent_external_results','query_clipped','action_clipped','evidence_clipped','joint_clipped',
 'query_tokens','action_tokens','evidence_tokens','observations',
 'literal_leaves','literal_in_user','literal_in_external','literal_in_assistant','literal_unmatched',
 'email_args','email_user_match','url_args','url_user_match','numeric_args','numeric_user_match',
 'last_external_failed','candidate_name_in_user','no_external_evidence','empty_args')
VIEW_NAMES = ('request','action','evidence','joint')

def producer(m):
    c=m.get('tool_call')
    return c.get('function') if isinstance(c,dict) else getattr(c,'function',None)

def leaves(value, path=''):
    if isinstance(value, dict):
        return [(p,v) for k,x in sorted(value.items()) for p,v in leaves(x,path+'/'+str(k))]
    if isinstance(value, (list,tuple)):
        return [(p,v) for i,x in enumerate(value) for p,v in leaves(x,path+'/'+str(i))]
    return [(path,value)]

def clip_text(tokenizer, text, limit):
    assert limit>=32,'Context view limit must preserve the truncation marker'
    ids=tokenizer.encode(text,add_special_tokens=False)
    if len(ids)<=limit: return text,False,len(ids)
    # Keep both ends. Selection is structural, never based on an evaluator label.
    left=limit//2;right=max(0,limit-left-8)
    out=tokenizer.decode(ids[:left])+ '\n[truncated]\n'+tokenizer.decode(ids[-right:] if right else [])
    while len(tokenizer.encode(out,add_special_tokens=False))>limit:
        if right:right=max(0,right-4)
        else:left=max(0,left-4)
        out=tokenizer.decode(ids[:left])+ '\n[truncated]\n'+tokenizer.decode(ids[-right:] if right else [])
    return out,True,len(ids)

def make_views(tokenizer, query, call, messages, description, external_tools, used):
    tools=[m for m in messages[:-1] if m.get('role')=='tool']
    ext=[m for m in tools if not producer(m) or producer(m) in external_tools]
    recent=ext[-3:]
    evidence='\n'.join(json.dumps({'producer':producer(m),'failed':bool(m.get('error')),
                           'observed':content(m.get('content'))},ensure_ascii=False) for m in recent)
    action=json.dumps({'tool':call.function,'description':description,'arguments':call.args},sort_keys=True,ensure_ascii=False)
    q,cq,nq=clip_text(tokenizer,query,480)
    a,ca,na=clip_text(tokenizer,action,480)
    e,ce,ne=clip_text(tokenizer,evidence,480)
    jq=clip_text(tokenizer,query,110)[0];ja=clip_text(tokenizer,action,180)[0];je=clip_text(tokenizer,evidence,180)[0]
    joint,cj,_=clip_text(tokenizer,'USER REQUEST\n'+jq+'\nPROPOSED ACTION\n'+ja+'\nOBSERVED TOOL DATA\n'+je,500)
    views=[q,a,e or '[no external observation]',joint]
    assert all(len(tokenizer.encode(t,add_special_tokens=False))<=512 for t in views)
    literal=[str(v) for _,v in leaves(call.args) if isinstance(v,(str,int,float)) and not isinstance(v,bool)]
    user=query.casefold();external='\n'.join(content(m.get('content')) for m in ext).casefold()
    assistant='\n'.join(content(m.get('content')) for m in messages[:-1] if m.get('role')=='assistant').casefold()
    denom=max(1,len(literal));emails=[v for v in literal if re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+',v)]
    urls=[v for v in literal if re.match(r'https?://',v)];nums=[v for v in literal if re.fullmatch(r'-?\d+(?:\.\d+)?',v)]
    oldcalls=[c for m in messages[:-1] if m.get('role')=='assistant' for c in m.get('tool_calls') or []]
    def signature(c):
        return digest({'tool':c.get('function'),'args':c.get('args')}) if isinstance(c,dict) else digest({'tool':c.function,'args':c.args})
    sig=signature(call);consecutive=0
    for c in reversed(oldcalls):
        if signature(c)!=sig:break
        consecutive+=1
    failed=0
    for m in reversed(tools):
        if not m.get('error'):break
        failed+=1
    turns=sum(bool(m.get('tool_calls')) for m in messages if m.get('role')=='assistant')
    extra=[float(can_review(messages,external_tools)),used/2,(2-used)/2,max(0,15-turns+1)/15,
     float(bool(tools and tools[-1].get('error'))),min(failed,15)/15,min(consecutive,15)/15,
     float(any(signature(c)==sig for c in oldcalls)),min(len(recent),3)/3,float(cq),float(ca),float(ce),float(cj or nq>110 or na>180 or ne>180),
     min(nq,32768)/32768,min(na,32768)/32768,min(ne,32768)/32768,min(len(tools),15)/15,
     min(len(literal),30)/30,sum(v.casefold() in user for v in literal)/denom,
     sum(v.casefold() in external for v in literal)/denom,sum(v.casefold() in assistant for v in literal)/denom,
     sum(v.casefold() not in user and v.casefold() not in external and v.casefold() not in assistant for v in literal)/denom,
     min(len(emails),10)/10,sum(v.casefold() in user for v in emails)/max(1,len(emails)),
     min(len(urls),10)/10,sum(v.casefold() in user for v in urls)/max(1,len(urls)),
     min(len(nums),10)/10,sum(v in query for v in nums)/max(1,len(nums)),
     float(bool(ext and ext[-1].get('error'))),float(call.function.casefold() in user),float(not ext),float(not call.args)]
    assert len(extra)==len(EXTRA_NAMES) and all(math.isfinite(v) for v in extra)
    return views,extra,dict(query_tokens=nq,action_tokens=na,evidence_tokens=ne,
        query_clipped=cq,action_clipped=ca,evidence_clipped=ce,source_role_matches_are_weak_features_not_authorization=True)
