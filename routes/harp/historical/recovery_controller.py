"""Single registered replay followed by the original untouched 15 tasks."""
import fcntl,hashlib,importlib.util,json,os,shutil,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('/root/autodl-tmp');OLD=BASE/'harp_ablation_20260930_setup2';ROOT=BASE/'harp_ablation_20261001_recovery1'
PY=str(BASE/'envs/promptkd_official_4090/bin/python');VARIANTS=('mid4_d32','last4_d16','last4_d64')
TASKS=[('caltech101','last4_d64')]+[(d,v) for d in ('stanford_cars','eurosat','ucf101','sun397','food101') for v in VARIANTS]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def original():
 sys.argv=[str(OLD/'run.py'),'diagnose','18645'];sp=importlib.util.spec_from_file_location('original_ablation',OLD/'run.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a);a.guard();return a
def guard():
 f=read(ROOT/'frozen_files.json')
 for p,h in f['files'].items():assert sha(Path(p))==h,('Frozen change',p)
 assert read(OLD/'status.json')['status']=='failed';return sha(ROOT/'frozen_files.json')
def prepare():
 assert not ROOT.exists();a=original();state=read(OLD/'status.json');done=[x for x in state['completed'] if x['action']=='evaluate'];assert len(done)==44
 assert (state['dataset'],state['variant'],state['seed'],state['action'])==('caltech101','last4_d64',2,'evaluate')
 files=dict(read(OLD/'frozen_files.json')['files']);reused=[]
 for t in done:
  p=OLD/f"seed_{t['seed']}"/t['dataset']/t['variant']/'harp';m=read(p/'audited_metrics.json')
  assert m['status']=='passed' and m['epochs']==20 and read(p/'full_training_audit.json')['passed']
  assert sha(p/'VLPromptLearner/model-best.pth.tar')==m['checkpoint_sha256']
  assert len((p/'epochs.jsonl').read_text().splitlines())==20
  for n in ('audited_metrics.json','epochs.jsonl','full_training_audit.json','VLPromptLearner/model-best.pth.tar'):files[str(p/n)]=sha(p/n)
  reused.append(dict(task=t,checkpoint_sha256=m['checkpoint_sha256']))
 files[str(OLD/'status.json')]=sha(OLD/'status.json');files[str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
 ROOT.mkdir();shutil.copyfile(OLD/'references.json',ROOT/'references.json');files[str(ROOT/'references.json')]=sha(ROOT/'references.json')
 dump(ROOT/'plan.json',dict(time=now(),source=str(OLD),reused=reused,tasks=TASKS,replay_budget=1,new_training_epochs=320,new_gpu_gate_steps=128,
  scientific_changes=[],hard_bounds_changed=False,old_failure_preserved=True,diagnostic_student_forwards_already=22,
  reason='旧queue无续排能力；本控制器固定SHA复用44项。旧Caltech失败具体算子根因尚未定位，单次复跑取证，不承诺已修复数值差异；复跑原硬检查全通过才自动接原15项，否则保留即停，不循环重试。'))
 files[str(ROOT/'plan.json')]=sha(ROOT/'plan.json');dump(ROOT/'frozen_files.json',dict(port=18645,time=now(),files=files));guard();print('PREPARED',len(reused),'reused',len(TASKS),'remaining',flush=True)
def worker(action,ds,var):
 guard();a=original();a.ROOT=ROOT
 # The inherited worker/guards/scientific functions remain unchanged.
 # Capture only evaluation tails of the one replay, with no changes to their arithmetic.
 if ds=='caltech101' and action in ('train','evaluate'):
  original_spec=importlib.util.spec_from_file_location
  def spec(name,path,*args,**kwargs):
   sp=original_spec(name,path,*args,**kwargs)
   if name=='inherited_science':
    execute=sp.loader.exec_module
    def exec_module(m):
     execute(m);base=m.build
     def build(*args,**kwargs):
      torch,tr=base(*args,**kwargs);dest=args[-1];active=[action=='evaluate'];current={};records=[]
      def observer(name):
       def hook(mod,inputs,z):
        z=z[0] if isinstance(z,tuple) else z
        if active[0] and not tr.model.training and isinstance(z,torch.Tensor) and z.ndim>=2 and ((z.shape[0] in (16,49)) or (z.ndim==3 and z.shape[1] in (16,49))):current[name]=z.detach().cpu().clone()
       return hook
      for name,mod in tr.model.named_modules():
       if name in ('image_encoder.conv1','image_encoder.ln_pre','image_encoder.ln_post','image_encoder','VPT_image_trans') or name.startswith('image_encoder.transformer.resblocks.') and name.count('.')==3:mod.register_forward_hook(observer(name))
      def end(mod,inputs,output):
       if active[0] and not tr.model.training and len(inputs[0]) in (16,49):
        feat,scale=output;records.append(dict(batch_size=len(inputs[0]),layers=dict(current),features=feat.detach().cpu().clone(),scale=scale.detach().cpu().clone(),input_hash=m.q.tensor_hash(inputs[0]),
         parameter_layout={n:dict(stride=list(p.stride()),offset=p.storage_offset(),alignment=p.data_ptr()%256) for n,p in mod.named_parameters()}));current.clear()
      tr.model.register_forward_hook(end)
      if action=='train':
       after=tr.after_train
       def recorded_after():
        active[0]=True
        try:return after()
        finally:torch.save(records,dest/'native_tail_trace.pt');active[0]=False
       tr.after_train=recorded_after
      else:
       test=tr.test
       def recorded_test(*args,**kwargs):
        try:return test(*args,**kwargs)
        finally:torch.save(records,dest/'reload_tail_trace.pt')
       tr.test=recorded_test
      return torch,tr
     m.build=build
    sp.loader.exec_module=exec_module
   return sp
  importlib.util.spec_from_file_location=spec
 try:a.worker(action,var,ds,2)
 finally:
  if ds=='caltech101' and action in ('train','evaluate'):importlib.util.spec_from_file_location=original_spec
def queue():
 guard();assert not (ROOT/'status.json').exists();lock=(BASE/'official_training_gpu0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(status='running',pid=os.getpid(),started=now(),completed=[],reused_runs=44,total_new_runs=16,scientific_changes=[],successor=None)
 dump(ROOT/'status.json',state)
 try:
  a=original()
  for ds,var in TASKS:
   dest=ROOT/'seed_2'/ds/var;dest.mkdir(parents=True,exist_ok=False);(dest/'r1').symlink_to(a.source(2,ds)/'r1',target_is_directory=True)
   for action in ('preflight','gate','train','evaluate'):
    guard();assert shutil.disk_usage(ROOT).free>=20*1024**3
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
    if action=='train':(dest/'harp').mkdir(exist_ok=False)
    state.update(action=action,dataset=ds,variant=var,seed=2,updated=now());dump(ROOT/'status.json',state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if action=='preflight' else '0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1');env.pop('PYTHONPATH',None);env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    with (dest/f'{action}.console.log').open('x') as log:subprocess.run([PY,str(Path(__file__).resolve()),action,ds,var],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    state['completed'].append(dict(action=action,dataset=ds,variant=var,seed=2));dump(ROOT/'status.json',state)
  state.update(status='completed_awaiting_analysis',finished=now(),action=None)
 except BaseException:state.update(status='failed',finished=now(),error=traceback.format_exc());raise
 finally:dump(ROOT/'status.json',state)
if __name__=='__main__':
 action=sys.argv[1]
 if action=='prepare':prepare()
 elif action=='queue':queue()
 else:worker(action,sys.argv[2],sys.argv[3])
