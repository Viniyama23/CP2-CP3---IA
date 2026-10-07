"""Fine-tuning QLoRA (4-bit) do Qwen2.5-1.5B-Instruct na GPU. Cabe em 8 GB de VRAM."""
import argparse, json, time, torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--out", default="outputs/run_default")
a = ap.parse_args()
assert torch.cuda.is_available(), "GPU CUDA não encontrada! Veja README (instalação do CUDA/PyTorch)."

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(
    a.model, device_map={"": 0}, torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

def encode(ex):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(msgs[-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + r)[:512]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ([-100] * len(p) + r)[:512]}

ds = load_dataset("json", data_files={"train": "data/train.jsonl", "test": "data/test.jsonl"})
ds = ds.map(encode, remove_columns=["messages"])

args = TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
    per_device_train_batch_size=4, gradient_accumulation_steps=4, per_device_eval_batch_size=4,
    bf16=True, optim="paged_adamw_8bit", lr_scheduler_type="cosine", warmup_ratio=0.05,
    logging_steps=5, eval_strategy="epoch", save_strategy="no", report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
    data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))

t0 = time.time(); trainer.train(); secs = time.time() - t0
ev = trainer.evaluate()
model.save_pretrained(a.out); tok.save_pretrained(a.out)
rec = {"run": a.out, "lr": a.lr, "lora_r": a.r, "epochs": a.epochs, "eval_loss": round(ev["eval_loss"], 4),
       "train_seconds": round(secs), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
print(rec)
with open("results/train_runs.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
