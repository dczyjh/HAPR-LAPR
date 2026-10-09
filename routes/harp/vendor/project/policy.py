"""Frozen added-group startup policy; no changes to common SGD parameters."""
import math

def effective_lr(epoch, batch, count, nominal):
    assert epoch >= 0 and count > 1 and 0 <= batch < count
    if epoch == 0:
        return 0.
    if epoch == 1:
        assert math.isclose(nominal, .0005, rel_tol=1e-12)
        return 1e-6 + (nominal-1e-6)*batch/(count-1)
    return nominal

def before_update(optimizer, epoch, batch, count):
    assert len(optimizer.param_groups)==2
    group=optimizer.param_groups[1]
    nominal=group['lr']
    assert math.isclose(nominal,optimizer.param_groups[0]['lr']*.1,rel_tol=1e-12)
    actual=effective_lr(epoch,batch,count,nominal)
    group['lr']=actual
    if epoch==0:
        for p in group['params']:
            assert not optimizer.state.get(p,{}),'Unexpected hold-phase momentum'
            p.grad=None
    return nominal,actual
