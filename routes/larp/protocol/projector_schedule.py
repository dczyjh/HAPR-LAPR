"""A single declared projector-only epoch-two ramp; SGD state is never reset."""
import math

def effective_lr(epoch,batch,count,nominal):
    assert epoch>=0 and count>1 and 0<=batch<count
    if epoch==1:
        assert math.isclose(nominal,.005,rel_tol=1e-12)
        return 1e-5+(.005-1e-5)*batch/(count-1)
    return nominal

def split_step(optimizer,original,projector_ids,lr,*args,**kwargs):
    """Temporarily separate parameter groups, preserving parameter/state identity."""
    assert not args and not kwargs,'No closure in this fixed SGD path'
    saved=optimizer.param_groups
    groups=[]
    for g in saved:
        prompt=[p for p in g['params'] if id(p) not in projector_ids]
        projector=[p for p in g['params'] if id(p) in projector_ids]
        if prompt:groups.append(dict(g,params=prompt))
        if projector:groups.append(dict(g,params=projector,lr=lr))
    assert sum(len(g['params']) for g in groups)==sum(len(g['params']) for g in saved)
    optimizer.param_groups=groups
    try:return original()
    finally:optimizer.param_groups=saved

def install(tr):
    params={n:p for n,p in tr.model.named_parameters() if p.requires_grad}
    ids={id(p) for n,p in params.items() if n.startswith('VPT_image_trans.')}
    assert len(ids)==6 and len(tr.optim.param_groups)==1
    original=tr.optim.step
    def step(*args,**kwargs):
        assert not args and not kwargs
        nominal=tr.optim.param_groups[0]['lr']
        actual=effective_lr(tr.epoch,tr.batch_idx,len(tr.train_loader_x),nominal)
        tr.projector_step_record=dict(epoch=tr.epoch+1,batch=tr.batch_idx+1,prompt_lr=nominal,projector_lr=actual)
        if tr.epoch==1:return split_step(tr.optim,original,ids,actual)
        return original()
    tr.optim.step=step
    return ids

def cpu_test(torch):
    rates=[effective_lr(1,i,368,.005) for i in range(368)]
    assert rates[0]==1e-5 and math.isclose(rates[-1],.005,rel_tol=1e-12)
    assert all(a<b for a,b in zip(rates,rates[1:]))
    for epoch in (0,2,19):assert effective_lr(epoch,0,368,.001)==.001
    for dtype in (torch.float32,torch.float64):
        a=torch.nn.Parameter(torch.tensor([2.,3.],dtype=dtype));b=torch.nn.Parameter(a.detach().clone())
        c=torch.nn.Parameter(a.detach().clone());e=torch.nn.Parameter(a.detach().clone())
        opt=torch.optim.SGD([a,b],lr=.005,momentum=.9,weight_decay=.0005)
        ref_a=torch.optim.SGD([c],lr=.005,momentum=.9,weight_decay=.0005)
        ref_b=torch.optim.SGD([e],lr=.005,momentum=.9,weight_decay=.0005)
        for i in range(368):
            for p in (a,b,c,e):p.grad=torch.ones_like(p)*.2
            ref_b.param_groups[0]['lr']=rates[i]
            split_step(opt,opt.step,{id(b)},rates[i]);ref_a.step();ref_b.step()
            assert torch.equal(a,c) and torch.equal(b,e)
            assert torch.equal(opt.state[a]['momentum_buffer'],ref_a.state[c]['momentum_buffer'])
            assert torch.equal(opt.state[b]['momentum_buffer'],ref_b.state[e]['momentum_buffer'])
            assert len(opt.param_groups)==1 and opt.param_groups[0]['lr']==.005
    return dict(status='passed',steps_per_epoch=368,first_lr=rates[0],last_lr=rates[-1],momentum_preserved=True)
