"""Compara modelo BASE x AJUSTADO no conjunto de teste (GPU). Métricas: ROUGE-L e F1 de tokens."""
import argparse, json, re, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", required=True)
ap.add_argument("--temperature", type=float, default=0.0)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
base = AutoModelForCausalLM.from_pretrained(a.model, device_map={"": 0}, torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
model = PeftModel.from_pretrained(base, a.adapter).eval()
words = lambda s: re.findall(r"\w+", s.lower())

def lcs(x, y):
    d = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(len(x)):
        for j in range(len(y)):
            d[i+1][j+1] = d[i][j] + 1 if x[i] == y[j] else max(d[i][j+1], d[i+1][j])
    return d[-1][-1]

def scores(pred, ref):
    p, r = words(pred), words(ref)
    if not p or not r: return 0.0, 0.0
    l = lcs(p, r); rl = 0 if l == 0 else 2 * (l/len(p)) * (l/len(r)) / (l/len(p) + l/len(r))
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    f1 = 0 if common == 0 else 2 * (common/len(p)) * (common/len(r)) / (common/len(p) + common/len(r))
    return rl, f1

def gen(msgs, use_adapter):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    kw = dict(do_sample=True, temperature=a.temperature) if a.temperature > 0 else dict(do_sample=False)
    with torch.no_grad():
        if use_adapter: out = model.generate(ids, max_new_tokens=300, **kw)
        else:
            with model.disable_adapter(): out = model.generate(ids, max_new_tokens=300, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

rows = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
res, agg = [], {"base": [0, 0], "ajustado": [0, 0]}
for r in rows:
    m = r["messages"]; item = {"pergunta": m[1]["content"], "referencia": m[2]["content"]}
    for name, flag in (("base", False), ("ajustado", True)):
        item[name] = gen(m[:2], flag); rl, f1 = scores(item[name], m[2]["content"])
        agg[name][0] += rl; agg[name][1] += f1
    res.append(item)
n = len(rows)
summary = {k: {"rouge_l": round(v[0]/n, 3), "f1_tokens": round(v[1]/n, 3)} for k, v in agg.items()}
print(json.dumps(summary, indent=2))
name = a.adapter.rstrip("/").split("/")[-1]
json.dump({"adapter": a.adapter, "resumo": summary, "amostras": res}, open(f"results/eval_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
