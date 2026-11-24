"""Minimal PPO (RLHF-style) fine-tuning of a small instruct LM to label MAUDE narratives.

Policy: causal LM + scalar value head. Reference: frozen copy for a per-token KL penalty.
Reward (terminal): F1 between predicted and FDA-coded failure modes, -0.5 for unparseable JSON.
Advantages: GAE over response tokens. Loss: clipped policy objective + clipped value loss.

Run on Kaggle (T4):  python ppo/ppo_train.py --data ppo_data.csv --model Qwen/Qwen2.5-0.5B-Instruct --out ppo_out
CPU smoke test:      python ppo/ppo_train.py --smoke
"""
import argparse, json, math, random, re, time
import torch, torch.nn as nn, torch.nn.functional as F

CATS = ["leak", "contamination", "breakage_material", "dosing_delivery", "mechanical_actuation",
        "labeling_packaging", "software_electronics", "use_interaction"]
PROMPT = ("Classify this FDA adverse-event report for an injection device. Allowed failure modes: "
          + ", ".join(CATS) + ".\nReport: {text}\nAnswer with JSON only, like {{\"failure_modes\": [\"leak\"]}}.\nJSON:")


def parse(s):
    m = re.search(r"\{.*?\}", s, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0)).get("failure_modes")
        return {x for x in v if x in CATS} if isinstance(v, list) else None
    except Exception:
        return None


def reward(text, gold):
    p = parse(text)
    if p is None:
        return -0.5
    if not p and not gold:
        return 1.0
    tp = len(p & gold)
    return 0.0 if tp == 0 else 2 * tp / (len(p) + len(gold))


class PolicyWithValue(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.v = nn.Linear(lm.config.hidden_size, 1)
        nn.init.zeros_(self.v.weight); nn.init.zeros_(self.v.bias)

    def forward(self, ids, mask):
        out = self.lm(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        return out.logits, self.v(out.hidden_states[-1].float()).squeeze(-1)


def token_logprobs(logits, ids):
    lp = F.log_softmax(logits[:, :-1].float(), -1)
    return lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, T-1], position t predicts token t+1


@torch.no_grad()
def generate(policy, tok, prompts, max_new, device, greedy=False):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=384).to(device)
    out = policy.lm.generate(**enc, max_new_tokens=max_new, do_sample=not greedy, temperature=1.0, top_k=0, top_p=1.0,
                             pad_token_id=tok.pad_token_id)
    P = enc.input_ids.shape[1]
    resp = out[:, P:]
    texts = tok.batch_decode(resp, skip_special_tokens=True)
    full_mask = torch.cat([enc.attention_mask, (resp != tok.pad_token_id).long()], 1)
    # keep the first EOS/pad inside the response as a real token so the episode has an end
    first_pad = (resp == tok.pad_token_id).long().argmax(1)
    for i, j in enumerate(first_pad.tolist()):
        if (resp[i] == tok.pad_token_id).any():
            full_mask[i, P + j] = 1
    return out, full_mask, P, texts


def evaluate(policy, tok, rows, device, bs=16, max_new=48):
    policy.eval(); r, valid = [], 0
    for i in range(0, len(rows), bs):
        b = rows[i:i + bs]
        _, _, _, texts = generate(policy, tok, [PROMPT.format(text=x["text"]) for x in b], max_new, device, greedy=True)
        for t, x in zip(texts, b):
            valid += parse(t) is not None
            r.append(max(reward(t, x["gold"]), 0.0))
    return {"mean_f1": round(sum(r) / len(r), 4), "valid_json": round(valid / len(rows), 4), "n": len(rows)}


def ppo(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    if args.smoke:
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers
        vocab = {c: i for i, c in enumerate(["<pad>"] + list("abcdefghijklmnopqrstuvwxyz_{}[]\":, .0123456789JSONRA:\n"))}
        tk = Tokenizer(models.WordLevel(vocab, unk_token="<pad>")); tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<pad>")
        mk = lambda: GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=512, n_embd=32, n_layer=2, n_head=2))
        torch.manual_seed(0); lm = mk(); torch.manual_seed(0); ref = mk()
        rows = [{"text": "leaking at hub", "gold": {"leak"}}, {"text": "foreign matter", "gold": {"contamination"}}] * 8
        train, test = rows, rows[:4]
        args.steps, args.batch, args.max_new = 2, 4, 8
    else:
        import pandas as pd
        tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float32
        lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device == "cuda" else dtype)
        lm.gradient_checkpointing_enable(); lm.config.use_cache = True
        df = pd.read_csv(args.data)
        conv = lambda d: [{"text": t, "gold": set(l.split("|"))} for t, l in zip(d.text, d.labels)]
        train, test = conv(df[df.split == "train"]), conv(df[df.split == "test"])
    tok.padding_side = "left"
    policy = PolicyWithValue(lm).to(device); ref.to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    log = {"config": vars(args), "before": evaluate(policy, tok, test, device, max_new=args.max_new), "steps": []}
    print("before", log["before"], flush=True)
    for step in range(args.steps):
        t0 = time.time(); policy.eval()
        batch = random.sample(train, args.batch)
        ids, mask, P, texts = generate(policy, tok, [PROMPT.format(text=x["text"]) for x in batch], args.max_new, device)
        R = torch.tensor([reward(t, x["gold"]) for t, x in zip(texts, batch)], device=device)
        with torch.no_grad():
            logits, values = policy(ids, mask)
            old_lp = token_logprobs(logits, ids)
            ref_lp = token_logprobs(ref(input_ids=ids, attention_mask=mask).logits, ids)
        rmask = mask[:, 1:].float().clone(); rmask[:, :P - 1] = 0  # response tokens only
        vals = values[:, :-1] * rmask
        kl = (old_lp - ref_lp) * rmask
        rew = -args.kl_coef * kl
        last = (rmask * torch.arange(rmask.shape[1], device=device)).argmax(1)
        rew[torch.arange(len(batch)), last] += R
        adv = torch.zeros_like(rew); gae = torch.zeros(len(batch), device=device)
        for t in reversed(range(rew.shape[1])):
            nv = vals[:, t + 1] if t + 1 < rew.shape[1] else torch.zeros_like(gae)
            delta = rew[:, t] + args.gamma * nv - vals[:, t]
            gae = (delta + args.gamma * args.lam * gae) * rmask[:, t]
            adv[:, t] = gae
        ret = adv + vals
        n = rmask.sum().clamp(min=1)
        mu = (adv * rmask).sum() / n; sd = (((adv - mu) ** 2 * rmask).sum() / n).sqrt()
        adv = (adv - mu) / (sd + 1e-8) * rmask
        policy.train(); stats = []
        for _ in range(args.ppo_epochs):
            for mb in torch.randperm(len(batch)).split(args.mb):
                lg, v = policy(ids[mb], mask[mb])
                lp = token_logprobs(lg, ids[mb]); m = rmask[mb]; mn = m.sum().clamp(min=1)
                ratio = torch.exp(lp - old_lp[mb])
                pg = torch.max(-adv[mb] * ratio, -adv[mb] * ratio.clamp(1 - args.clip, 1 + args.clip))
                v = v[:, :-1]; vc = vals[mb] + (v - vals[mb]).clamp(-args.clip_v, args.clip_v)
                vl = 0.5 * torch.max((v - ret[mb]) ** 2, (vc - ret[mb]) ** 2)
                loss = (pg * m).sum() / mn + args.vf_coef * (vl * m).sum() / mn
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
                stats.append({"loss": loss.item(), "clipfrac": (((ratio - 1).abs() > args.clip).float() * m).sum().item() / mn.item()})
        s = {"step": step, "reward": round(R.mean().item(), 4), "kl": round((kl.sum(1)).mean().item(), 4),
             "valid_json": round(sum(parse(t) is not None for t in texts) / len(texts), 3),
             "loss": round(sum(x["loss"] for x in stats) / len(stats), 4),
             "clipfrac": round(sum(x["clipfrac"] for x in stats) / len(stats), 4), "sec": round(time.time() - t0, 1)}
        log["steps"].append(s); print(s, flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            log.setdefault("mid", []).append({"step": step + 1, **evaluate(policy, tok, test, device, max_new=args.max_new)})
            print("eval", log["mid"][-1], flush=True)
    log["after"] = evaluate(policy, tok, test, device, max_new=args.max_new)
    print("after", log["after"], flush=True)
    if args.out:
        import os
        os.makedirs(args.out, exist_ok=True)
        json.dump(log, open(f"{args.out}/ppo_log.json", "w"), indent=1)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mb", type=int, default=4); ap.add_argument("--ppo_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--clip_v", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.1); ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--eval_every", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ppo(ap.parse_args())
