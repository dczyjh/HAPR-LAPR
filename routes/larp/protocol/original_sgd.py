"""Exact retained install() body, injected dependencies, no training entry point."""
import math

def install(torch,tr,m):
    params={n:v for n,v in tr.model.named_parameters() if v.requires_grad}
    projector={id(v) for n,v in params.items() if n.startswith('VPT_image_trans.')}
    added=[v for n,v in params.items() if q.added(n)]
    assert len(projector)==6 and len(tr.optim.param_groups)==(2 if m=='larp' else 1)
    assert bool(added)==(m=='larp')
    if added:assert {id(v) for v in added}=={id(v) for v in tr.optim.param_groups[1]['params']}
    original=tr.optim.step
    def step(*args,**kwargs):
        assert not args and not kwargs
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in params.values())
        scheduled=[g['lr'] for g in tr.optim.param_groups];e,b=tr.epoch,tr.batch_idx
        actual=f.ps.effective_lr(e,b,f.COUNT,scheduled[0])
        if added:assert math.isclose(scheduled[1],.1*scheduled[0],rel_tol=1e-12)
        first=[v.detach().clone() for v in added] if e==0 and b==0 else None
        result=f.ps.split_step(tr.optim,original,projector,actual) if e==1 else original()
        assert [g['lr'] for g in tr.optim.param_groups]==scheduled
        assert all(torch.isfinite(v).all() for v in params.values())
        assert all('momentum_buffer' in tr.optim.state[v] for v in added)
        if first is not None and added:assert any(not torch.equal(a,v) for a,v in zip(first,added)),'Added group must update from first batch'
        tr.actual_step=dict(epoch=e+1,batch=b+1,scheduled_lr=scheduled,prompt_lr=scheduled[0],
            projector_lr=actual,added_lr=scheduled[1] if added else None,added_held=False)
        return result
    tr.optim.step=step
