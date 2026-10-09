"""Bounded HARP structure ablations; inherit frozen training and hard evaluation."""
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp')
PYTHON = str(BASE/'envs/promptkd_official_4090/bin/python')
DATASETS = ('fgvc_aircraft', 'dtd', 'oxford_pets', 'oxford_flowers', 'caltech101',
            'stanford_cars', 'eurosat', 'ucf101', 'sun397', 'food101')
VARIANTS = {'mid4_d32': ([4,5,6,7],32), 'last4_d16': ([8,9,10,11],16),
            'last4_d64': ([8,9,10,11],64)}
PORT = int(sys.argv[2])
assert PORT in (18645,26006)
SEEDS = (1,2) if PORT==18645 else (3,)

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def source(seed,dataset):
    if PORT==26006: return BASE/'harp_ten_seed3_26006_20260930_setup2'/f'seed_{seed}'/dataset
    if seed==1:
        return BASE/'harp_pets_full_pair_20260929' if dataset=='oxford_pets' else BASE/'harp_ten_seed1_20260929'/dataset
    if dataset=='eurosat': return BASE/'harp_eurosat_seed2_replay_20260930'/'seed_2'/dataset
    return BASE/'harp_ten_seed23_20260929_setup2'/f'seed_{seed}'/dataset
def science_source():
    return BASE/('harp_ten_seed23_20260929_setup2' if PORT==18645 else 'harp_ten_seed3_26006_20260930_setup2')
def guard():
    f=read(ROOT/'frozen_files.json');assert f['port']==PORT
    for p,h in f['files'].items(): assert sha(p)==h,('Frozen change',p)
    return sha(ROOT/'frozen_files.json')
def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()

def prepare():
    assert not (ROOT/'frozen_files.json').exists()
    assert not (ROOT/'status.json').exists()
    import ast
    ast.parse((ROOT/'run.py').read_text())
    files={str(ROOT/n):sha(ROOT/n) for n in ('run.py','研究登记.md','实现与结果口径.md')}
    for p in science_source().iterdir():
        if p.suffix=='.py' or p.name in ('frozen_files.json','references.json','source_attestation.json'):
            files[str(p)]=sha(p)
    refs={}
    for seed in SEEDS:
        for dataset in DATASETS:
            src=source(seed,dataset);pair={}
            for method in ('r1','harp'):
                d=src/method;a=read(d/'audited_metrics.json')
                assert a['status']=='passed' and a['epochs']==20 and a['seed']==seed and a['dataset']==dataset
                assert a['server_port']==PORT
                assert a['backend']==dict(benchmark=True,deterministic=False,deterministic_algorithms=False,flash=True,efficient=False,math=True)
                assert a['common_schedule']=='projector_epoch2_ramp'
                assert read(d/'full_training_audit.json')['passed']
                assert sha(d/'VLPromptLearner/model-best.pth.tar')==a['checkpoint_sha256']
                assert len((d/'epochs.jsonl').read_text().splitlines())==20
                for n in ('audited_metrics.json','metrics.json','initial_shared.json','initial_trainable.pt','batches.jsonl','epochs.jsonl','resolved.yaml','full_training_audit.json'):
                    files[str(d/n)]=sha(d/n)
                pair[method]=dict(path=str(d),checkpoint_sha256=a['checkpoint_sha256'],metrics=a)
            assert read(src/'r1/metrics.json')['identity']==read(src/'harp/metrics.json')['identity']
            assert read(src/'r1/initial_shared.json')==read(src/'harp/initial_shared.json')
            for n in ('audit.json',):
                p=src/'gate_r1'/n;files[str(p)]=sha(p)
            assert read(src/'gate_r1/audit.json')['status']=='passed'
            refs[f'{seed}/{dataset}']=pair
    dump(ROOT/'references.json',refs)
    files[str(ROOT/'references.json')]=sha(ROOT/'references.json')
    dump(ROOT/'frozen_files.json',dict(port=PORT,time=now(),files=files))
    print('FROZEN',PORT,len(refs),'pairs',len(files),'files',flush=True)

def worker(action,variant,dataset,seed):
    assert variant in VARIANTS and dataset in DATASETS and seed in SEEDS
    guard();src=source(seed,dataset)
    dest=ROOT/f'seed_{seed}'/dataset/variant
    # Import an unchanged frozen runner. Its original source guards remain active.
    saved=list(sys.argv)
    sys.argv=[saved[0],action,'harp',dataset,str(2 if PORT==18645 else 3)]
    sp=importlib.util.spec_from_file_location('inherited_science',science_source()/'run.py')
    m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
    m.SEEDS=SEEDS;m.SEED=seed;m.ROOT=dest
    sys.argv[4]=str(seed)
    inherited_guard=m.guard;original_config=m.config;original_build=m.build
    q=m.q
    def checked_guard():
        inherited_guard();return guard()
    m.guard=checked_guard
    layers,width=VARIANTS[variant]
    def config(train,root,method,ds,sd,epochs,out):
        cfg,args=original_config(train,root,method,ds,sd,epochs,out)
        if method=='harp':
            cfg.defrost();cfg.TRAINER.PROMPTKD.ADAPTATION.LAYERS=layers
            cfg.TRAINER.PROMPTKD.ADAPTATION.HARP_DIM=width;cfg.freeze()
            args.opts+=['TRAINER.PROMPTKD.ADAPTATION.LAYERS',str(layers),'TRAINER.PROMPTKD.ADAPTATION.HARP_DIM',str(width)]
        return cfg,args
    q.config=config
    holder={}
    def build(method,ds,sd,epochs,out):
        torch,tr=original_build(method,ds,sd,epochs,out)
        holder.update(torch=torch,tr=tr)
        if method=='r1':
            assert not any(q.added(n) for n,p in tr.model.named_parameters())
            return torch,tr
        adapters={n:p for n,p in tr.model.named_parameters() if q.added(n)}
        assert sum(p.numel() for p in adapters.values())==4*(1537*width+768)
        assert all(p.requires_grad and p.dtype==torch.float16 for p in adapters.values())
        positions=sorted({int(n.split('resblocks.')[1].split('.')[0]) for n in adapters})
        assert positions==layers
        prior=torch.load(src/'harp/initial_trainable.pt',map_location='cpu')
        for n,p in adapters.items():
            if '.up.' in n or n.endswith('.down.bias'): assert torch.count_nonzero(p)==0
            if variant=='mid4_d32':
                idx=int(n.split('resblocks.')[1].split('.')[0])
                old=n.replace(f'resblocks.{idx}.',f'resblocks.{idx+4}.')
                assert torch.equal(p.detach().cpu(),prior[old]),('Layer-mapped initialization',n)
        q.dump(out/'ablation_identity.json',dict(time=now(),variant=variant,port=PORT,dataset=ds,seed=sd,
            layers=layers,width=width,added_parameters=sum(p.numel() for p in adapters.values()),
            reference_harp=str(src/'harp'),reference_r1=str(src/'r1'),frozen=guard(),
            initialization='exact_layer_mapping' if variant=='mid4_d32' else 'same_rule_different_shape',
            scientific_changes=['LAYERS'] if variant=='mid4_d32' else ['HARP_DIM']))
        return torch,tr
    m.build=build;q.build=build
    if action=='preflight':
        m.preflight()  # Original common schedule, data, assets and environment checks.
        torch,train,project=q.framework('r1')
        import yaml
        default,_=original_config(train,project,'harp',dataset,seed,20,dest/'harp')
        changed,_=config(train,project,'harp',dataset,seed,20,dest/'harp')
        a=yaml.safe_load(default.dump());b=yaml.safe_load(changed.dump())
        check=copy.deepcopy(b);check['TRAINER']['PROMPTKD']['ADAPTATION'].update(LAYERS=[8,9,10,11],HARP_DIM=32)
        assert a==check
        from trainers.efficient_adaptation import HighLevelAdapter
        with torch.random.fork_rng(devices=[]):
            x=HighLevelAdapter(768,width,.001,torch.float32)
            assert sum(p.numel() for p in x.parameters())*4==4*(1537*width+768)
            assert torch.count_nonzero(x(torch.ones(2,3,768)))==0
        q.dump(dest/'variant_preflight.json',dict(status='passed',time=now(),original=a,variant=b,
            changed_keys=['LAYERS'] if variant=='mid4_d32' else ['HARP_DIM'],frozen=guard()))
    elif action=='gate':
        assert read(dest/'variant_preflight.json')['status']=='passed'
        m.gate('r1')
        m.gate('harp')
    elif action=='train':
        assert read(dest/'gate_harp/audit.json')['status']=='passed'
        assert read(dest/'gate_r1/audit.json')['status']=='passed'
        assert not (dest/'harp/initial_trainable.pt').exists()
        try:q.train_run('harp',dataset,seed,20,dest/'harp')
        except BaseException:
            if holder:
                torch,tr=holder['torch'],holder['tr']
                import random,numpy as np
                torch.save(dict(model=tr.model.state_dict(),optimizer=tr.optim.state_dict(),scheduler=tr.sched.state_dict(),
                    batch=tr.failure_batch[0],epoch=tr.epoch,batch_idx=tr.batch_idx,
                    gradients={n:None if p.grad is None else p.grad.detach().cpu() for n,p in tr.model.named_parameters() if p.requires_grad},
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),dest/'harp/first_failure_state.pt')
            raise
    elif action=='evaluate':
        assert not (dest/'harp/canonical_val_evidence.pt').exists()
        m.evaluate('harp',dataset,seed,20,dest/'harp')
        report=read(dest/'harp/audited_metrics.json')
        report.update(experiment_id=ROOT.name,result_role='registered_structural_ablation',variant=variant,
                      protocol_version='1.65-harp',server_port=PORT,reference_harp=str(src/'harp'),reference_r1=str(src/'r1'))
        dump(dest/'harp/audited_metrics.json',report)
    else:raise ValueError(action)

def queue():
    guard();assert not (ROOT/'status.json').exists()
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=(BASE/'official_training_gpu0.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',started=now(),pid=os.getpid(),port=PORT,completed=[],
               total_runs=len(SEEDS)*len(DATASETS)*len(VARIANTS),successor=None)
    dump(ROOT/'status.json',state)
    try:
        for seed in SEEDS:
            for dataset in DATASETS:
                for variant in VARIANTS:
                    dest=ROOT/f'seed_{seed}'/dataset/variant;dest.mkdir(parents=True,exist_ok=False)
                    (dest/'r1').symlink_to(source(seed,dataset)/'r1',target_is_directory=True)
                    for action in ('preflight','gate','train','evaluate'):
                        guard();assert shutil.disk_usage(ROOT).free>=20*1024**3
                        idle=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                        assert not idle,('GPU busy',idle)
                        if action=='train':(dest/'harp').mkdir(exist_ok=False)
                        state.update(action=action,variant=variant,dataset=dataset,seed=seed,updated=now());dump(ROOT/'status.json',state)
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if action=='preflight' else '0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
                        env.pop('PYTHONPATH',None);env.pop('CUBLAS_WORKSPACE_CONFIG',None)
                        with (dest/f'{action}.console.log').open('x') as out:
                            subprocess.run([PYTHON,str(ROOT/'run.py'),action,str(PORT),variant,dataset,str(seed)],env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
                        state['completed'].append(dict(action=action,variant=variant,dataset=dataset,seed=seed));dump(ROOT/'status.json',state)
        state.update(status='completed_awaiting_analysis',finished=now(),action=None)
    except BaseException:
        state.update(status='failed',finished=now(),error=traceback.format_exc());raise
    finally:dump(ROOT/'status.json',state)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='queue':queue()
    else:worker(action,sys.argv[3],sys.argv[4],int(sys.argv[5]))
