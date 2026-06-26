#!/usr/bin/env python3
"""Reproduce the core of AQLM training (no block fine-tuning) at a kernel-friendly
config: per-output scale + BEAM-SEARCH code assignment + least-squares codebook update,
alternating. This is what greedy residual VQ lacked. If beam search + LSQ beats greedy
(6.32) and the scalar 4-bit codebook (6.34) at M=4, the additive scheme our kernel
decodes finally wins on accuracy at equal bits.

Run:  python llama_aqlm_train.py
"""
import torch, torch.nn as nn, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "NousResearch/Llama-2-7b-hf"
K = 256; D = 8; SEQLEN = 2048; PPL_SAMPLES = 30
MC = int(os.environ.get("MC", "4"))      # number of additive codebooks (M)
ROUNDS = int(os.environ.get("ROUNDS", "1")); BEAM = 4
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def kmeans(P, k=K, iters=5, chunk=1_000_000):
    N = P.shape[0]; C = P[torch.randperm(N, device=P.device)[:k]].clone()
    asg = torch.zeros(N, dtype=torch.long, device=P.device)
    for _ in range(iters):
        Cn2 = (C*C).sum(1)
        for s in range(0, N, chunk):
            p = P[s:s+chunk]; asg[s:s+chunk] = ((p*p).sum(1,keepdim=True) - 2*(p@C.t()) + Cn2).argmin(1)
        Cnew = torch.zeros_like(C); cnt = torch.zeros(k, device=P.device)
        Cnew.index_add_(0, asg, P); cnt.index_add_(0, asg, torch.ones(N, device=P.device)); C = Cnew/cnt.clamp(min=1).unsqueeze(1)
    return C

@torch.no_grad()
def beam_search(W, C, B=BEAM, chunk=40000):
    N, d = W.shape; Mc = C.shape[0]
    codes = torch.empty(N, Mc, dtype=torch.long, device=W.device)
    for s in range(0, N, chunk):
        w = W[s:s+chunk]; n = w.shape[0]
        d0 = ((w[:,None,:] - C[0][None,:,:])**2).sum(-1)            # (n,K)
        _, idx = d0.topk(B, dim=1, largest=False)
        bc = idx.unsqueeze(-1); br = C[0][idx]                     # (n,B,1),(n,B,d)
        for m in range(1, Mc):
            cand = br[:,:,None,:] + C[m][None,None,:,:]            # (n,B,K,d)
            sc = ((w[:,None,None,:] - cand)**2).sum(-1).reshape(n, B*C.shape[1])
            _, flat = sc.topk(B, dim=1, largest=False)
            bsel = flat // C.shape[1]; ksel = flat % C.shape[1]
            bc = torch.cat([torch.gather(bc, 1, bsel.unsqueeze(-1).expand(-1,-1,m)), ksel.unsqueeze(-1)], -1)
            br = torch.gather(br, 1, bsel.unsqueeze(-1).expand(-1,-1,d)) + C[m][ksel]
        codes[s:s+chunk] = bc[:,0,:]
    return codes

@torch.no_grad()
def lsq_update(W, codes, Mc, reg=1e-2):
    N, d = W.shape; P = Mc*K
    feat = codes + (torch.arange(Mc, device=W.device)*K)[None,:]   # (N,Mc)
    AtW = torch.zeros(P, d, device=W.device)
    for m in range(Mc): AtW.index_add_(0, feat[:,m], W)
    AtA = torch.zeros(P, P, device=W.device); ones = torch.ones(N, device=W.device)
    for m in range(Mc):
        for mp in range(Mc):
            AtA.view(-1).index_add_(0, feat[:,m]*P + feat[:,mp], ones)
    AtA += reg*torch.eye(P, device=W.device)
    return torch.linalg.solve(AtA, AtW).reshape(Mc, K, d)

@torch.no_grad()
def aqlm_quantize(W, Mc):
    OC, IC = W.shape
    s = W.abs().amax(1, keepdim=True).clamp(min=1e-8)
    P = (W/s).reshape(OC, IC//D, D).reshape(-1, D).contiguous()     # (N,d)
    R = P.clone(); cb = []                                           # residual k-means init
    for m in range(Mc):
        cm = kmeans(R, K); cb.append(cm)
        codes_m = beam_search(R, cm.unsqueeze(0))[:,0]; R = R - cm[codes_m]
    C = torch.stack(cb)
    for _ in range(ROUNDS):
        codes = beam_search(P, C)        # (N,Mc)
        C = lsq_update(P, codes, Mc)
    codes = beam_search(P, C)
    recon = sum(C[m][codes[:,m]] for m in range(Mc))
    return (recon.reshape(OC, IC//D, D).reshape(OC, IC)) * s

@torch.no_grad()
def quantize_model(model, Mc):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
            mod.weight.data = aqlm_quantize(mod.weight.data.float(), Mc).to(mod.weight.dtype); torch.cuda.empty_cache()

@torch.no_grad()
def ppl(model, enc):
    nseq = min(PPL_SAMPLES, enc.size(1)//SEQLEN); lf = nn.CrossEntropyLoss(reduction="sum"); tot=0.0; nt=0
    for i in range(nseq):
        ids = enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device); o = model(ids).logits
        tot += lf(o[:, :-1, :].reshape(-1, o.size(-1)).float(), ids[:, 1:].reshape(-1)).item(); nt += ids[:,1:].numel()
    return float(torch.tensor(tot/nt).exp())

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()
    print("fp16 PPL = %.4f" % ppl(model, enc), flush=True)
    t0 = time.time(); quantize_model(model, MC)
    bits = MC*8/D
    print("AQLM-trained M=%d (%.0f-bit, beam+LSQ): PPL = %.4f  (%.0fs)" % (MC, bits, ppl(model, enc), time.time()-t0), flush=True)
    print("(refs: fp16 5.83 | scalar 4-bit 6.34 | greedy-additive M=4 6.32 | AQLM-real-2x8 7.63)", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
