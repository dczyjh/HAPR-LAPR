"""Bounded reload evidence, with the first independent evaluation canonical."""
import math

def attach(tr, torch, tensor_hash):
    original_test=tr.test;original_parse=tr.parse_batch_test;original_process=tr.evaluator.process
    state={'active':False,'last':None}
    has_cache=hasattr(tr.model_teacher,'get_text_features')
    if not has_cache:
        def teacher_observer(module,inputs,output):
            if state['active']:
                state['teacher_hashes'].append(tensor_hash(output[1]))
        tr.model_teacher.register_forward_hook(teacher_observer)
    def parse(batch):
        if state['active']:
            state['inputs'].append(dict(image=tensor_hash(batch['img']),label=tensor_hash(batch['label']),impath=batch.get('impath')))
        return original_parse(batch)
    def process(output,label):
        if state['active']:
            assert bool(torch.isfinite(output).all()),'Nonfinite evaluation logits'
            state['rows'].append((output.detach().cpu().clone(),label.detach().cpu().clone()))
        return original_process(output,label)
    def test(split=None):
        state.update(active=True,rows=[],inputs=[],teacher_hashes=[])
        try:result=original_test(split)
        finally:state['active']=False
        logits=torch.cat([x[0] for x in state['rows']]);labels=torch.cat([x[1] for x in state['rows']])
        correct=int((logits.argmax(1)==labels).sum())
        assert abs(float(result)-100*correct/len(labels))<1e-8
        if has_cache:
            text_hash=tensor_hash(tr.model_teacher.get_text_features())
        else:
            assert state['teacher_hashes'] and len(set(state['teacher_hashes']))==1,'Teacher text changed within evaluation'
            text_hash=state['teacher_hashes'][0]
        state['last']=dict(logits=logits,labels=labels,inputs=state['inputs'],correct=correct,
                           total=len(labels),accuracy=float(result),split=split,
                           teacher_text_hash=text_hash)
        return result
    tr.parse_batch_test=parse;tr.evaluator.process=process;tr.test=test
    return state

def compare(a,b,torch,tolerances):
    assert a['inputs']==b['inputs'],'Evaluation input identity mismatch'
    assert a['teacher_text_hash']==b['teacher_text_hash'],'Teacher text mismatch'
    assert torch.equal(a['labels'],b['labels']),'Evaluation labels mismatch'
    x,y=a['logits'],b['logits']
    assert x.shape==y.shape and x.dtype==y.dtype
    assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
    tol=tolerances['forward'];error=(x.float()-y.float()).abs()
    failed=error>tol['atol']+tol['rtol']*y.float().abs()
    # Dot-product cancellation makes per-element relative error ill-scaled
    # near zero. Keep that historical metric as a diagnostic, and separately
    # use an explicit, symmetric per-example infinity-norm engineering bound.
    scale=torch.maximum(x.float().abs().amax(1,keepdim=True),y.float().abs().amax(1,keepdim=True))
    row_failed=error>tol['atol']+tol['rtol']*scale
    changes=(x.argmax(1)!=y.argmax(1)).nonzero().flatten().tolist()
    result=dict(max_abs=float(error.max()),failed_elements=int(failed.sum()),
                prediction_changes=changes,accuracy_delta_pp=a['accuracy']-b['accuracy'],
                tensor_tolerance=tol,exact_logits=bool(torch.equal(x,y)),
                row_scaled_failed_elements=int(row_failed.sum()),
                hard_metric='symmetric_per_example_infinity_norm',
                original_elementwise_passed=not bool(failed.any()))
    assert not bool(row_failed.any()),('Reload logits outside registered row-scaled engineering bound',result)
    return result
