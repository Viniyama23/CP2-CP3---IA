"""Frontend (Gradio): digite o prompt e compare modelo base x ajustado. Uso: python src/app.py --adapter outputs/<run>"""
import argparse, torch, gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", required=True)
a = ap.parse_args()
SYSTEM = "Você é um assistente de suporte de TI (helpdesk nível 1). Responda em português, em passos numerados, curtos e objetivos."

tok = AutoTokenizer.from_pretrained(a.model)
base = AutoModelForCausalLM.from_pretrained(a.model, device_map={"": 0}, torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
model = PeftModel.from_pretrained(base, a.adapter).eval()

def responder(prompt, ajustado, temperature, top_p, max_tokens):
    ids = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt").to("cuda")
    kw = dict(max_new_tokens=int(max_tokens), do_sample=temperature > 0, temperature=max(temperature, 0.01), top_p=top_p)
    with torch.no_grad():
        if ajustado: out = model.generate(ids, **kw)
        else:
            with model.disable_adapter(): out = model.generate(ids, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

gr.Interface(fn=responder, title="Assistente de Suporte de TI (LLM local na GPU)",
    inputs=[gr.Textbox(label="Seu problema", lines=3), gr.Checkbox(True, label="Usar modelo ajustado (desmarque p/ base)"),
            gr.Slider(0, 1.5, 0.3, label="temperature"), gr.Slider(0.1, 1, 0.9, label="top_p"),
            gr.Slider(50, 500, 300, step=10, label="max_new_tokens")],
    outputs=gr.Textbox(label="Resposta", lines=12)).launch()
