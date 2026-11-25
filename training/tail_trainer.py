import json,time,pathlib,shutil
import torch
from transformers import Trainer,TrainerCallback
from supervised_tail_loss_candidate import supervised_tail_loss

class TailTrainer(Trainer):
    def __init__(self,*args,trace_path=None,**kwargs):
        super().__init__(*args,**kwargs)
        # Each microbatch has one example with its own mean assistant-token loss.
        # Trainer must divide by the accumulation count, not infer normalization
        # from Qwen's **kwargs signature (which ignores num_items_in_batch).
        self.model_accepts_loss_kwargs=False
        self.trace_path=pathlib.Path(trace_path) if trace_path else None
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        clean={k:v for k,v in inputs.items() if not k.startswith('_')}
        loss,output=supervised_tail_loss(model,clean)
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite assistant loss')
        return (loss,output) if return_outputs else loss
    def training_step(self,model,inputs,num_items_in_batch=None):
        cuda=self.args.device.type=='cuda'
        if cuda:torch.cuda.synchronize()
        start=time.perf_counter()
        loss=super().training_step(model,inputs,num_items_in_batch)
        if cuda:torch.cuda.synchronize()
        if self.trace_path:
            record={'sample_id':inputs.get('_sample_id'),'optimizer_step_before':self.state.global_step,'forward_backward_seconds':time.perf_counter()-start,'cpu_preprocessing_seconds':inputs.get('_preprocessing_seconds'),'input_tokens':int(inputs['input_ids'].shape[-1]),'supervised_tokens':int((inputs['labels']!=-100).sum()),'returned_accumulation_scaled_loss':float(loss.detach())}
            with self.trace_path.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        return loss

class FiniteAndDiskGuard(TrainerCallback):
    def __init__(self,out):self.out=pathlib.Path(out)
    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        for n,p in model.named_parameters():
            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError('Non-finite gradient: '+n)
    def on_optimizer_step(self,args,state,control,model=None,**kwargs):
        if any(p.requires_grad and not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Non-finite adapter parameter')
    def on_step_end(self,args,state,control,**kwargs):
        if control.should_save and shutil.disk_usage(self.out).free<2*1024**3:
            raise RuntimeError('Less than2GiB disk free before checkpoint; stopping safely')
        (self.out/'live_progress.json').write_text(json.dumps({'global_step':state.global_step,'epoch':state.epoch,'time':time.time()}))
