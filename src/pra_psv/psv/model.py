"""Train-only semantic PCA + small interaction ensemble; two PRA-conditioned heads."""
import collections, math, time
from .core import SEEDS, LABELS

def tensor_pack(packs, preprocessing):
    import torch
    mean,basis=preprocessing['embedding_mean'],preprocessing['basis']
    x=torch.tensor([p['views'] for p in packs],dtype=torch.float32)
    x=torch.nn.functional.normalize(x,dim=-1)
    z=(x-mean)@basis
    numerical=torch.tensor([p['numeric'] for p in packs],dtype=torch.float32)
    # Explicit request-action and action-observation interactions.
    x=torch.cat([z.flatten(1),z[:,0]*z[:,1],z[:,1]*z[:,2],numerical],dim=-1)
    x=(x-preprocessing['mean'])/preprocessing['scale']
    outlier=x.abs().amax(dim=-1)>8
    return x.clamp(-8,8),outlier

def network(dim):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(dim,32),torch.nn.GELU(),torch.nn.Dropout(.15),
                               torch.nn.Linear(32,16),torch.nn.GELU(),torch.nn.Linear(16,6),torch.nn.Tanh())

def conditioned(output, blocked):
    return output.reshape(-1,2,3)[__import__('torch').arange(len(blocked)),blocked.long()]

def fit(rows, phase):
    import torch
    torch.set_num_threads(4)
    train=[r for r in rows if r['split']=='train' and r['targets'] is not None]
    val=[r for r in rows if r['split']=='validation_select' and r['targets'] is not None]
    assert not {r['group'] for r in train}&{r['group'] for r in val}
    if len(train)<24 or len(val)<8: raise ValueError('Insufficient known train/selection pairs')
    delta=lambda r:r['targets'][0]+2*r['targets'][1]
    if sum(delta(r)!=0 for r in train)<5 or sum(delta(r)!=0 for r in val)<2:
        raise ValueError('Insufficient nonzero train/selection preference support; do not claim routing ranking verified')
    torch.manual_seed(20261024)
    allviews=torch.tensor([r['pack']['views'] for r in train],dtype=torch.float32).reshape(-1,3584)
    allviews=torch.nn.functional.normalize(allviews,dim=-1)
    mean=allviews.mean(0);q=min(32,len(allviews)-1)
    _,_,basis=torch.pca_lowrank(allviews-mean,q=q,center=False,niter=2)
    preprocessing=dict(embedding_mean=mean,basis=basis)
    # Construct unstandardized training design once; no selection/test input fitting.
    z=(allviews.reshape(len(train),4,3584)-mean)@basis
    design=torch.cat([z.flatten(1),z[:,0]*z[:,1],z[:,1]*z[:,2],torch.tensor([r['pack']['numeric'] for r in train])],dim=-1)
    scale=design.std(0,unbiased=False);scale[scale<.001]=1
    preprocessing.update(mean=design.mean(0),scale=scale)
    x,_=tensor_pack([r['pack'] for r in train],preprocessing)
    vx,vood=tensor_pack([r['pack'] for r in val],preprocessing)
    y=torch.tensor([r['targets'] for r in train]);vy=torch.tensor([r['targets'] for r in val])
    blocked=torch.tensor([r['pack']['pra_block'] for r in train]);vb=torch.tensor([r['pack']['pra_block'] for r in val])
    groups=collections.Counter(r['group'] for r in train)
    weights=torch.tensor([1/groups[r['group']] for r in train]);weights/=weights.mean()
    # Cap preference reweighting; retain zero-benefit and harmful interventions.
    weights*=torch.tensor([2. if delta(r)!=0 else 1. for r in train]);weights/=weights.mean()
    states=[];choices=[];start=time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed);model=network(x.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.03)
        best=None
        for epoch in range(1,61):
            model.train();pred=conditioned(model(x),blocked)
            loss=((pred-y).square().mean(-1)*weights).mean()
            advantage=y[:,0]+2*y[:,1];pvalue=pred[:,0]+2*pred[:,1];mask=advantage!=0
            if mask.any():loss+=.15*torch.nn.functional.softplus(-advantage[mask].sign()*pvalue[mask]).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            model.eval()
            with torch.no_grad():
                vp=conditioned(model(vx),vb);vl=float((vp-vy).square().mean())
            if best is None or vl<best[0]:best=(vl,epoch,{k:v.detach().clone() for k,v in model.state_dict().items()})
        states.append(best[2]);choices.append(dict(seed=seed,epoch=best[1],selection_loss=best[0]))
    support={}
    for b in (False,True):
        tr=[r for r in train if r['pack']['pra_block']==b];vr=[r for r in val if r['pack']['pra_block']==b]
        support[str(b)]=dict(train=len(tr),selection=len(vr),train_nonzero=sum(delta(r)!=0 for r in tr),
            selection_nonzero=sum(delta(r)!=0 for r in vr),train_help=sum(delta(r)>0 for r in tr),train_harm=sum(delta(r)<0 for r in tr))
        support[str(b)]['enabled']=len(tr)>=8 and support[str(b)]['train_nonzero']>=2 and support[str(b)]['selection_nonzero']>=1
    package=dict(format='psv_action_router_ensemble_v1',phase=phase,preprocessing=preprocessing,
                 dim=x.shape[1],states=states,support=support,choices=choices,labels=LABELS)
    with torch.no_grad():
        preds=[]
        for state in states:
            net=network(x.shape[1]);net.load_state_dict(state);net.eval();preds.append(conditioned(net(vx),vb))
        ensemble=torch.stack(preds);values=ensemble[:,:,0]+2*ensemble[:,:,1]
        uncertainty=values.std(0,unbiased=False)
        package['uncertainty_cutoff']=max(.05,float(torch.quantile(uncertainty,.8)))
    # Fixed preference rule: positive safe-completion/attack-prevention value;
    # selection inputs set the same nominal 30% review fraction as the old pilot.
    package['route_cutoff']=max(0.,float(torch.quantile(values.mean(0),.7)))
    package['report']=dict(train=len(train),selection=len(val),unknown=sum(r['targets'] is None for r in rows),
        train_help=sum(delta(r)>0 for r in train),train_harm=sum(delta(r)<0 for r in train),
        selection_help=sum(delta(r)>0 for r in val),selection_harm=sum(delta(r)<0 for r in val),
        selection_nonzero=sum(delta(r)!=0 for r in val),support=support,choices=choices,
        training_seconds=time.perf_counter()-start,PCA_fit_on_train_only=True,
        trained_on_current_test=False,uncertainty_is_not_calibrated_probability=True,
        OOD_selection=sum(vood.tolist()))
    return package

class Router:
    def __init__(self,package):
        import torch
        self.package=package;self.models=[]
        for state in package['states']:
            n=network(package['dim']);n.load_state_dict(state);n.eval();self.models.append(n)
    def predict(self,pack):
        import torch
        x,ood=tensor_pack([pack],self.package['preprocessing']);b=torch.tensor([pack['pra_block']])
        with torch.inference_mode():p=torch.stack([conditioned(n(x),b)[0] for n in self.models])
        value=p[:,0]+2*p[:,1];mu=float(value.mean());std=float(value.std(unbiased=False))
        supported=self.package['support'][str(pack['pra_block'])]['enabled']
        abstain=not supported or bool(ood[0]) or std>self.package['uncertainty_cutoff']
        return dict(value=mu,std=std,safe_gain=float(p[:,0].mean()),attack_prevention=float(p[:,1].mean()),
            utility_gain=float(p[:,2].mean()),abstain=abstain,unsupported_branch=not supported,
            out_of_support=bool(ood[0]),selected=not abstain and mu>0 and mu>=self.package['route_cutoff'])
