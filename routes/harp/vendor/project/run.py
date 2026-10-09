"""Freeze the seed-1 HARP protocol and run ten paired datasets at seeds 2/3."""
import copy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
MASTER=Path(__file__).resolve().parent
DATASETS=('caltech101','fgvc_aircraft','dtd','oxford_flowers','stanford_cars','eurosat','ucf101','sun397','oxford_pets','food101')
SEEDS=(2,3)
DATASET=sys.argv[3] if len(sys.argv)>3 else DATASETS[0]
SEED=int(sys.argv[4]) if len(sys.argv)>4 else SEEDS[0]
assert DATASET in DATASETS
assert SEED in SEEDS
ROOT=MASTER/f'seed_{SEED}'/DATASET
TRACE=Path('/root/autodl-tmp/pets_paired_trace_20260928')
STARTUP=Path('/root/autodl-tmp/harp_aircraft_startup_20260929')
OLD=Path('/root/autodl-tmp/backend_pilot_20260928_fix2')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
ev=load('evaluation_evidence',MASTER/'evaluation_evidence.py')
ps=load('fixed_projector_schedule',MASTER/'projector_schedule.py')
ss=load('fixed_added_schedule',STARTUP/'policy.py')
q=load('full_pair_queue',OLD/'research_queue.py')
base_guard=q.guard;base_config=q.config;base_build=q.build
METHODS=('r1','harp');COUNT=q.SPECS[DATASET][0]//8
def probe_path(dataset,seed,method):
    assert dataset==DATASET and seed==SEED and method in METHODS
    return ROOT/('gate_'+method)
q.probe_path=probe_path

def guard():
    base_guard()
    for p,h in q.read(MASTER/'references.json').items():assert q.sha(Path(p))==h,p
    for p,h in q.read(MASTER/'frozen_files.json').items():assert q.sha(MASTER/p)==h,p
    return q.sha(MASTER/'frozen_files.json')

def config(train,root,method,dataset,seed,epochs,out):
    assert method in METHODS and dataset==DATASET and seed==SEED and epochs==20
    original_seeds=q.SEEDS
    try:
        q.SEEDS=SEEDS
        cfg,args=base_config(train,root,method,dataset,seed,epochs,out)
    finally:
        q.SEEDS=original_seeds
    if method=='harp':
        cfg.defrost();cfg.TRAINER.PROMPTKD.ADAPTATION.LR_MULT=.1;cfg.freeze()
        args.opts+=['TRAINER.PROMPTKD.ADAPTATION.LR_MULT','0.1']
    return cfg,args
q.config=config

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

def build(method,dataset,seed,epochs,out):
    guard()
    torch,tr=base_build(method,dataset,seed,epochs,out)
    assert Path(sys.modules['evaluation_evidence'].__file__).resolve()==MASTER/'evaluation_evidence.py'
    action=sys.argv[1]
    q.dump(out/(action+'_overlay.json'),dict(protocol='1.54',experiment_id=MASTER.name,dataset=DATASET,seed=SEED,frozen=guard(),
        common='projector_epoch2_1e-5_to_.005',added='hold1_ramp2_then_0.1' if method=='harp' else None,
        source_projector=q.sha(MASTER/'projector_schedule.py'),source_added=q.sha(STARTUP/'policy.py')))
    if action not in ('train','gate'):return torch,tr
    install(torch,tr,method)
    if action=='gate':return torch,tr
    pair=None
    if method=='harp':
        pair=[json.loads(x) for x in (ROOT/'r1/batches.jsonl').read_text().splitlines()]
        assert len(pair)==20*COUNT
        assert q.read(ROOT/'r1/audited_metrics.json')['status']=='passed'
    old_fb=tr.forward_backward;means=[0.]*20;counts=[0]*20
    last_batch=[None]
    teacher={n:q.tensor_hash(v) for n,v in tr.model_teacher.state_dict().items()}
    def fb(batch):
        last_batch[0]=batch;e,b=tr.epoch,tr.batch_idx;i=e*COUNT+b
        ident=dict(image=q.tensor_hash(batch['img']),label=q.tensor_hash(batch['label']),impath=batch.get('impath'))
        if pair is not None:assert ident==pair[i]['identity'],'Full R1/HARP input pairing'
        assert tr.num_batches==COUNT
        if e==1 and b==0:
            import random,numpy as np
            torch.save(dict(model=tr.model.state_dict(),optimizer=tr.optim.state_dict(),scheduler=tr.sched.state_dict(),
                batch=batch,epoch=e,batch_idx=b,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'epoch2_batch1_pre.pt')
        result=old_fb(batch)
        assert all(math.isfinite(float(v)) for v in result.values())
        row=dict(tr.actual_step,identity=ident,loss=float(result['loss']),loss_kd=float(result.get('loss_kd',result['loss'])))
        with (out/'batches.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        counts[e]+=1;means[e]+=row['loss_kd'];return result
    tr.forward_backward=fb
    old_test=tr.test;last_base=[None]
    def test(split=None):
        val=old_test(split)
        if split=='val':last_base[0]=float(val)
        return val
    tr.test=test;old_epoch=tr.after_epoch
    def after_epoch():
        old_epoch();e=tr.epoch
        assert counts[e]==COUNT and last_base[0] is not None
        assert shutil.disk_usage(ROOT).free>=20*1024**3,'Disk below hard reserve'
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=e+1,mean_kd=means[e]/COUNT,base=last_base[0]))+'\n')
        if e==2:tr.save_model(e,str(out),val_result=last_base[0],model_name='diagnostic-epoch3.pth.tar')
    tr.after_epoch=after_epoch
    original_after_train=tr.after_train
    def after_train():
        original_after_train()
        assert counts==[COUNT]*20
        assert teacher=={n:q.tensor_hash(v) for n,v in tr.model_teacher.state_dict().items()}
        q.dump(out/'full_training_audit.json',dict(passed=True,batches=sum(counts),teacher_unchanged=True))
    tr.after_train=after_train
    tr.failure_batch=last_batch
    return torch,tr
q.build=build

def preflight():
    assert not (ROOT/'preflight.json').exists(),'No preflight overwrite'
    frozen=guard();torch,train,root=q.framework('r1');assert not torch.cuda.is_available()
    assert q.read(MASTER/'seed1_source_audit.json')['status']=='passed'
    import yaml
    from dassl.optim import build_optimizer,build_lr_scheduler
    from trainers.adaptation_optim import adaptation_param_groups
    from types import SimpleNamespace
    class Tiny(torch.nn.Module):
        def __init__(self,method):
            super().__init__();self.VPT=torch.nn.Parameter(torch.ones(1))
            self.VPT_image_trans=torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(6)])
            if method=='harp':
                self.harp_adapter=torch.nn.Linear(1,1)
    all_rates={};configs={};initials=[]
    for method in METHODS:
        cfg,args=config(train,root,method,DATASET,SEED,20,ROOT/method)
        configs[method]=yaml.safe_load(cfg.dump())
        model=Tiny(method);opt=build_optimizer(model,cfg.OPTIM,param_groups=adaptation_param_groups(model,cfg))
        sched=build_lr_scheduler(opt,cfg.OPTIM)
        fake=SimpleNamespace(model=model,optim=opt,epoch=0,batch_idx=0)
        install(torch,fake,method)
        all_rates[method]=[]
        for epoch in range(20):
            fake.epoch=epoch;all_rates[method].append([g['lr'] for g in opt.param_groups])
            for batch in range(COUNT if epoch<2 else 1):
                fake.batch_idx=batch
                for p in model.parameters():p.grad=torch.ones_like(p)*.1
                opt.step()
                assert fake.actual_step['projector_lr']==ps.effective_lr(epoch,batch,COUNT,all_rates[method][-1][0])
            sched.step()
        p=Path('/root/autodl-tmp/harp_research_20260928/bridge')/DATASET/f'seed_{SEED}'/method;meta=q.read(p/'probe.json')
        assert meta['status']=='passed' and q.sha(p/'probe.pt')==meta['evidence_sha256']
        evidence=torch.load(p/'probe.pt',map_location='cpu')
        # Old True probes supply initialization/input identity, not new-backend equivalence.
        assert evidence['backend']['benchmark'] is True
        initials.append({k:evidence[k] for k in ('initial_shared','rng','classnames','batches')})
        q.asset_check(cfg,root);q.data_check(cfg,DATASET)
    assert initials[0]==initials[1]
    a,b=[copy.deepcopy(configs[m]) for m in METHODS]
    b['OUTPUT_DIR']=a['OUTPUT_DIR'];b['TRAINER']['PROMPTKD']['ADAPTATION'].update(TYPE='none',LR_MULT=1.)
    assert a==b,'Unexpected common configuration difference'
    assert [x[0] for x in all_rates['r1']]==[x[0] for x in all_rates['harp']]
    ps.cpu_test(torch)
    q.dump(ROOT/'preflight.json',dict(status='passed',time=q.now(),frozen=frozen,configs=configs,
        lr_trajectories=all_rates,initial_pairing=True,environment=q.old_queue().environment_guard(),decode=q.old_queue().decode_guard()))

def gate(method):
    out=ROOT/('gate_'+method);out.mkdir(exist_ok=False)
    torch,tr=build(method,DATASET,SEED,20,out)
    frozen=q.common_hashes(tr.model,True);teacher={n:q.tensor_hash(v) for n,v in tr.model_teacher.state_dict().items()}
    expected=torch.load(Path('/root/autodl-tmp/harp_research_20260928/bridge')/DATASET/f'seed_{SEED}'/method/'probe.pt',map_location='cpu')
    assert q.common_hashes(tr.model)==expected['initial_shared']
    rng={'cpu':q.tensor_hash(torch.get_rng_state()),'cuda':q.tensor_hash(torch.cuda.get_rng_state())}
    assert rng==expected['rng']
    iterator=iter(tr.train_loader_x);batch=next(iterator);second=next(iterator)
    tr.num_batches=COUNT;tr.set_model_mode('train');records=[]
    ident=dict(image=q.tensor_hash(batch['img']),label=q.tensor_hash(batch['label']),impath=batch.get('impath'))
    assert ident==expected['batches'][0]
    second_ident=dict(image=q.tensor_hash(second['img']),label=q.tensor_hash(second['label']),impath=second.get('impath'))
    assert second_ident==expected['batches'][1]
    initialization=dict(initial_shared=q.common_hashes(tr.model),rng=rng,backend=q.backend(torch),
        classnames=tr.dm.dataset.classnames,batches=[ident,second_ident])
    if method=='harp':
        paired=torch.load(ROOT/'gate_r1/probe.pt',map_location='cpu')
        assert initialization==paired,'New backend gate input/initialization pairing'
    rates=q.read(ROOT/'preflight.json')['lr_trajectories'][method]
    for e,b in ((0,0),(1,0),(1,COUNT-1),(2,0)):
        tr.epoch=e;tr.batch_idx=b
        for g,lr in zip(tr.optim.param_groups,rates[e]):g['lr']=lr
        loss=tr.forward_backward(batch);assert all(math.isfinite(float(v)) for v in loss.values())
        records.append(dict(tr.actual_step,loss=loss))
    assert frozen==q.common_hashes(tr.model,True)
    assert teacher=={n:q.tensor_hash(v) for n,v in tr.model_teacher.state_dict().items()}
    before={n:q.tensor_hash(v) for n,v in tr.model.state_dict().items()}
    tr.save_model(2,str(out),val_result=0.,model_name='model-best.pth.tar');tr.load_model(str(out))
    assert before=={n:q.tensor_hash(v) for n,v in tr.model.state_dict().items()}
    tr.close_writer();q.dump(out/'audit.json',dict(status='passed',time=q.now(),frozen=guard(),steps=records,
        no_evaluation=True,synthetic_boundaries_not_training_result=True,checkpoint_exact=True))
    torch.save(initialization,out/'probe.pt')
    q.dump(out/'probe.json',dict(status='passed',evidence_sha256=q.sha(out/'probe.pt'),
        scope='initialization_inputs_and_four_boundary_steps_not_original_numerical_bridge',backend=q.backend(torch)))

def queue():
    lock=(MASTER/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=Path('/root/autodl-tmp/official_training_gpu0.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (MASTER/'status.json').exists()
    guard();assert q.read(MASTER/'seed1_source_audit.json')['status']=='passed'
    state=dict(status='running',started=q.now(),pid=os.getpid(),completed=[],successor=None,
        seeds=list(SEEDS),new_formal_runs=40);q.dump(MASTER/'status.json',state)
    try:
        for seed in SEEDS:
            for dataset in DATASETS:
                directory=MASTER/f'seed_{seed}'/dataset;directory.mkdir(parents=True,exist_ok=False)
                jobs=[('preflight','r1')]+[('gate',m) for m in METHODS]+[(a,m) for m in METHODS for a in ('train','evaluate')]
                for action,method in jobs:
                    guard();q.old_queue().ensure_gpu_idle();assert shutil.disk_usage(MASTER).free>=20*1024**3
                    if action=='train':(directory/method).mkdir(exist_ok=False)
                    state.update(action=action,method=method,dataset=dataset,seed=seed,updated=q.now());q.dump(MASTER/'status.json',state)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if action=='preflight' else '0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
                    env.pop('PYTHONPATH',None);env.pop('CUBLAS_WORKSPACE_CONFIG',None)
                    with (directory/f'{method}_{action}.console.log').open('x') as f:
                        subprocess.run([q.PYTHON,str(MASTER/'run.py'),action,method,dataset,str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                    state['completed'].append(dict(action=action,method=method,dataset=dataset,seed=seed));q.dump(MASTER/'status.json',state)
        state.update(status='completed_awaiting_analysis',finished=q.now(),action=None,method=None)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),finished=q.now());raise
    finally:q.dump(MASTER/'status.json',state)

read,dump,sha,now=q.read,q.dump,q.sha,q.now
common_hashes,added,canonical=q.common_hashes,q.added,q.canonical
tensor_hash,backend=q.tensor_hash,q.backend
TOLERANCES,SPECS,SETTINGS,ROUTE,IMPLEMENTATION=q.TOLERANCES,q.SPECS,q.SETTINGS,q.ROUTE,q.IMPLEMENTATION
def check_update(name,momentum,changed,held,torch):
    if held and added(name):
        assert momentum is None and changed==0, ('Hold-phase checkpoint mismatch',name)
    else:
        assert momentum is not None and torch.isfinite(momentum).all(),name


def evaluate(method,dataset,seed,epochs,out):
    torch,tr = build(method,dataset,seed,epochs,out)
    initial = torch.load(out/'initial_trainable.pt',map_location='cpu')
    assert read(out/'initial_shared.json')==common_hashes(tr.model)
    frozen = common_hashes(tr.model,True)
    audits = {}
    best = None
    for tag,name in (('best','model-best.pth.tar'),('last',f'model.pth.tar-{epochs}')):
        checkpoint = torch.load(out/'VLPromptLearner'/name,map_location='cpu')
        assert 1 <= checkpoint['epoch'] <= epochs
        if tag=='last':
            assert checkpoint['epoch']==epochs
        else:
            best = checkpoint
        assert all(not torch.is_floating_point(x) or torch.isfinite(x).all() for x in checkpoint['state_dict'].values())
        assert all(tensor_hash(checkpoint['state_dict'][n])==frozen[canonical(n)]
                   for n,p in tr.model.named_parameters() if not p.requires_grad and not added(n))
        updates = {n:int(torch.count_nonzero(checkpoint['state_dict'][n]-x)) for n,x in initial.items()}
        held = checkpoint['epoch']==1
        rows = []
        names = {id(p):n for n,p in tr.model.named_parameters()}
        for saved,live in zip(checkpoint['optimizer']['param_groups'],tr.optim.param_groups):
            assert len(saved['params'])==len(live['params'])
            for key,p in zip(saved['params'],live['params']):
                if not p.requires_grad:
                    continue
                n = names[id(p)]
                momentum = checkpoint['optimizer']['state'].get(key,{}).get('momentum_buffer')
                check_update(n,momentum,updates[n],held,torch)
                rows.append({'name':n,'momentum_nonzero':0 if momentum is None else int(torch.count_nonzero(momentum)),
                             'changed_elements':updates[n]})
        assert {r['name'] for r in rows}==set(initial)
        for group in ('visual_prompt','projector') + (() if held or method.startswith('r0') or method=='r1' else ('adaptation',)):
            predicate = (lambda n:added(n)) if group=='adaptation' else (
                (lambda n:n.startswith('VPT_image_trans.')) if group=='projector' else
                (lambda n:n.startswith('image_encoder.') and not added(n)))
            assert any(r['changed_elements'] and r['momentum_nonzero'] for r in rows if predicate(r['name'])), group
        audits[tag] = rows
    tr.load_model(str(out))
    assert tr.model.state_dict().keys()==best['state_dict'].keys()
    assert all(torch.equal(v.detach().cpu(),best['state_dict'][n]) for n,v in tr.model.state_dict().items())
    tr.epoch = best['epoch']-1
    evidence=None
    if not method.startswith('r0'):
        from evaluation_evidence import attach,compare
        evidence=attach(tr,torch,tensor_hash)
    b = float(tr.test(split='val'))
    bc = [tr.evaluator._total,tr.evaluator._correct]
    base_evidence=evidence['last'] if evidence else None
    n = float(tr.test(split='test'))
    nc = [tr.evaluator._total,tr.evaluator._correct]
    assert (bc[0],nc[0])==SPECS[dataset][1:3]
    assert abs(b-100*bc[1]/bc[0])<1e-8 and abs(n-100*nc[1]/nc[0])<1e-8
    assert math.isfinite(b+n) and 0<=b<=100 and 0<=n<=100
    checkpoint_sha=sha(out/'VLPromptLearner/model-best.pth.tar')
    torch.save(dict(base_evidence,checkpoint_sha256=checkpoint_sha),out/'canonical_val_evidence.pt')
    torch.save(dict(evidence['last'],checkpoint_sha256=checkpoint_sha),out/'canonical_test_evidence.pt')
    reload_audit={}
    if method.startswith('r0'):
        assert abs(b-float(best['val_result'])) < 1e-8, ('Reload mismatch',b,best['val_result'])
    if not method.startswith('r0'):
        native = read(out/'metrics.json')
        assert native['method']==('none' if method=='r1' else method)
        assert native['seed']==seed and native['epochs']==epochs and native['selection']=='best_val'
        assert native['selected_epoch']==best['epoch']
        checkpoint_sha=sha(out/'VLPromptLearner/model-best.pth.tar')
        for label,record,canonical_accuracy in (('val',base_evidence,b),('test',evidence['last'],n)):
            prior=torch.load(out/f'native_{label}_evidence.pt',map_location='cpu')
            assert prior['checkpoint_sha256']==checkpoint_sha
            key='base' if label=='val' else 'novel'
            assert abs(prior['accuracy']-native[key])<1e-8
            torch.save(dict(first=record,reference=prior,tolerances=TOLERANCES),out/f'reload_pair_native_{label}.pt')
            reload_audit['native_'+label]=compare(record,prior,torch,TOLERANCES)
            torch.save(dict(record,checkpoint_sha256=checkpoint_sha),out/f'canonical_{label}_evidence.pt')
        selected=torch.load(out/'best_val_evidence.pt',map_location='cpu')
        assert selected['checkpoint_sha256']==checkpoint_sha
        assert abs(selected['accuracy']-float(best['val_result']))<1e-8
        torch.save(dict(first=base_evidence,reference=selected,tolerances=TOLERANCES),out/'reload_pair_selection_val.pt')
        reload_audit['selection_val']=compare(base_evidence,selected,torch,TOLERANCES)
    report = dict(status='passed',time=now(),method=method,dataset=dataset,seed=seed,epochs=epochs,
        is_formal=epochs==20,base=b,novel=n,hm=2*b*n/(b+n) if b+n else 0.,
        selected_epoch=best['epoch'],kd=SPECS[dataset][3],base_reload_delta=b-float(best['val_result']),
        counts_base_novel=[bc,nc],checkpoint_sha256=sha(out/'VLPromptLearner/model-best.pth.tar'),
        audits=audits,frozen=guard(),server_port=SETTINGS['port'],route=ROUTE,
        baseline_identity='R1@'+str(SETTINGS['port']),implementation=IMPLEMENTATION,
        backend=backend(torch),protocol_version='1.54',reload_audit=reload_audit,
        canonical_evaluation='first_independent_reload_no_selection',
        numerical_policy='diagnostic_only_user_authorized')
    report.update(experiment_id=MASTER.name,result_role='exploratory_paired_full20',common_schedule='projector_epoch2_ramp',added_schedule='hold1_ramp2_then_0.1' if method=='harp' else None)
    epochs_rows=[json.loads(x) for x in (out/'epochs.jsonl').read_text().splitlines()]
    assert len(epochs_rows)==20 and report['selected_epoch']==max(epochs_rows,key=lambda x:x['base'])['epoch']
    assert read(out/'full_training_audit.json')['passed']
    if method=='harp':
        assert read(out/'metrics.json')['identity']==read(ROOT/'r1/metrics.json')['identity']
        assert read(out/'initial_shared.json')==read(ROOT/'r1/initial_shared.json')
    dump(out/'audited_metrics.json',report)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='preflight':preflight()
    elif action=='queue':queue()
    elif action=='gate':gate(sys.argv[2])
    elif action=='evaluate':
        out=ROOT/sys.argv[2];assert not (out/'canonical_val_evidence.pt').exists()
        evaluate(sys.argv[2],DATASET,SEED,20,out)
    elif action=='train':
        for m in METHODS:assert q.read(ROOT/('gate_'+m)/'audit.json')['status']=='passed'
        method=sys.argv[2];out=ROOT/method
        assert not (out/'initial_trainable.pt').exists()
        # Capture the live trainer for failure preservation without changing train_run.
        holder={};wrapped_build=q.build
        def capture(*args,**kwargs):
            torch,tr=wrapped_build(*args,**kwargs);holder.update(torch=torch,tr=tr);return torch,tr
        q.build=capture
        try:q.train_run(method,DATASET,SEED,20,out)
        except BaseException:
            if holder:
                torch,tr=holder['torch'],holder['tr']
                import random,numpy as np
                torch.save(dict(model=tr.model.state_dict(),optimizer=tr.optim.state_dict(),scheduler=tr.sched.state_dict(),
                    batch=tr.failure_batch[0],epoch=tr.epoch,batch_idx=tr.batch_idx,
                    gradients={n:None if p.grad is None else p.grad.detach().cpu() for n,p in tr.model.named_parameters() if p.requires_grad},
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'first_failure_state.pt')
            raise
    else:raise SystemExit('Unknown action')
