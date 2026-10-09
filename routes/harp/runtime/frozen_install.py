"""Verbatim frozen scientific functions; see provenance/scientific_functions.json."""
from types import SimpleNamespace
import policy as ss
import projector_schedule as ps

def added(name):
    return any(x in name for x in ('harp_adapter','lora_A_','lora_B_'))

q = SimpleNamespace(added=added)
COUNT = None

def install(torch,tr,method):
    params={n:p for n,p in tr.model.named_parameters() if p.requires_grad}
    projector={id(p) for n,p in params.items() if n.startswith('VPT_image_trans.')}
    assert len(projector)==6 and len(tr.optim.param_groups)==(2 if method=='harp' else 1)
    added=[p for n,p in params.items() if q.added(n)]
    assert bool(added)==(method=='harp')
    if method=='harp':assert {id(p) for p in added}=={id(p) for p in tr.optim.param_groups[1]['params']}
    held=[p.detach().clone() for p in added]
    original=tr.optim.step
    def step(*args,**kwargs):
        assert not args and not kwargs
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params.values()),'Nonfinite raw gradient'
        scheduled=[g['lr'] for g in tr.optim.param_groups]
        e,b=tr.epoch,tr.batch_idx
        actual=ps.effective_lr(e,b,COUNT,scheduled[0])
        added_lr=None
        if method=='harp':
            nominal,added_lr=ss.before_update(tr.optim,e,b,COUNT)
            assert nominal==scheduled[1]
        try:
            if e==1:result=ps.split_step(tr.optim,original,projector,actual)
            else:result=original()
        finally:
            if method=='harp':tr.optim.param_groups[1]['lr']=scheduled[1]
        assert [g['lr'] for g in tr.optim.param_groups]==scheduled
        if e==0:
            for p,v in zip(added,held):assert torch.equal(p,v) and not tr.optim.state.get(p,{})
        assert all(torch.isfinite(p).all() for p in params.values()),'Nonfinite updated parameter'
        tr.actual_step=dict(epoch=e+1,batch=b+1,scheduled_lr=scheduled,prompt_lr=scheduled[0],
            projector_lr=actual,added_lr=added_lr,added_held=e==0 and method=='harp')
        return result
    tr.optim.step=step
