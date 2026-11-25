"""Candidate memory optimization, NOT enabled in formal train_lora.py yet.

Qwen3-VL supports logits_to_keep. Prompt/video labels are -100, so vocabulary
projection and cross entropy for those positions are unnecessary. Transformer
inputs, video frames, attention, base precision and assistant targets are identical.
Requires full Qwen video loss/gradient comparison on GPU before formal adoption.
"""
def supervised_tail_loss(model, inputs):
    import torch
    import torch.nn.functional as F
    labels=inputs['labels']
    assert labels.ndim==2 and labels.shape[0]==1
    active=(labels[0]!=-100).nonzero().flatten()
    assert active.numel()>0
    start=int(active[0]);n=int(active.numel())
    assert start>=1 and start+n==labels.shape[1]
    assert torch.equal(active,torch.arange(start,labels.shape[1],device=active.device))
    kwargs={k:v for k,v in inputs.items() if k!='labels'}
    output=model(**kwargs,logits_to_keep=n+1)
    # Kept hidden positions: [start-1, ..., L-1]. The last predicts beyond EOS
    # and is unused, matching the normal left-shifted causal-LM objective.
    logits=output.logits[:,:-1,:].contiguous().float()
    target=labels[:,start:].contiguous()
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))
    return loss,output

def self_test():
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace
    torch.manual_seed(20261102)
    results=[]
    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embed=torch.nn.Embedding(37,13);self.head=torch.nn.Linear(13,37)
        def forward(self,input_ids,logits_to_keep=0,**kwargs):
            hidden=self.embed(input_ids)
            # Cumulative context makes target gradients also flow to prompt inputs.
            hidden=hidden.cumsum(1)/torch.arange(1,hidden.shape[1]+1).view(1,-1,1)
            return SimpleNamespace(logits=self.head(hidden[:,-logits_to_keep:,:]))
    for L,N in [(7,1),(17,5),(53,28)]:
        model=SmallModel();ids=torch.randint(0,37,(1,L));labels=ids.clone();labels[:,:L-N]=-100
        full=model(input_ids=ids).logits
        baseline=F.cross_entropy(full[:,:-1,:].contiguous().float().reshape(-1,37),labels[:,1:].reshape(-1),ignore_index=-100)
        baseline.backward();grads={k:v.grad.detach().clone() for k,v in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        actual,_=supervised_tail_loss(model,{'input_ids':ids,'labels':labels})
        actual.backward()
        assert torch.allclose(baseline,actual,atol=1e-6,rtol=1e-6)
        maxdiff=max(float((grads[k]-v.grad).abs().max()) for k,v in model.named_parameters())
        assert all(torch.allclose(grads[k],v.grad,atol=1e-6,rtol=1e-5) for k,v in model.named_parameters())
        results.append({'L':L,'target_tokens':N,'baseline_loss':float(baseline.detach()),'candidate_loss':float(actual.detach()),'max_gradient_difference':maxdiff})
    return {'status':'synthetic_cpu_loss_and_gradient_equivalence_passed','cases':results,'limitations':'Toy contextual model only; actual Qwen3-VL video GPU equivalence and memory peak remain untested. Formal training code unchanged.'}

if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
