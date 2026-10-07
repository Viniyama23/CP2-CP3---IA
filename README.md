# CP2/CP3 — LLM local para Suporte de TI (Helpdesk) com fine-tuning na GPU

**Grupo:** _(nomes)_ | **Entrega:** 15/10/2026 | **GPU:** NVIDIA RTX 3060 Ti 8 GB (ASUS)

## 1. Modelo escolhido
**Qwen2.5-1.5B-Instruct** (Alibaba, licença Apache-2.0). Motivos: bom em português, pequeno o bastante para ajustar com **QLoRA 4-bit em 8 GB de VRAM**, e vem com *chat template*.

## 2. Problema escolhido
Assistente de **suporte de TI nível 1 (helpdesk)**: dado um problema (internet, VPN, impressora, senha, phishing...), responder em **passos numerados curtos e objetivos, em PT-BR**. O modelo base tende a responder de forma longa/genérica ou misturando idiomas; o ajuste ensina o padrão de resposta.

Dataset (`src/make_dataset.py`): 13 problemas × 2 formas de perguntar × 5 variações de contexto = 130 exemplos de treino. Teste: uma **3ª forma de perguntar, nunca vista no treino** (26 exemplos).
> Limitação (cite na apresentação): as respostas-alvo são as mesmas por tema, então o teste mede generalização da *pergunta*, não conhecimento novo. Para melhorar, ampliem o dataset com casos reais do grupo.

## 3. Parâmetros testados
`run_experiments.sh` treina 4 configurações (learning rate {1e-4, 2e-4} × rank LoRA {8, 16}), 3 épocas cada. Resultados em `results/train_runs.jsonl` (eval_loss, tempo, VRAM de pico). Sugestão extra para o grupo: variar `temperature` (0, 0.3, 0.8) no `evaluate.py --temperature`.

## 4. Avaliação
`src/evaluate.py` compara **base × ajustado** no teste com ROUGE-L e F1 de tokens, salva as respostas em `results/eval_<run>.json`. Além das métricas, **leiam as amostras** e comentem erros qualitativos.

| Config | eval_loss | ROUGE-L base | ROUGE-L ajustado |
|---|---|---|---|
| lr1e-4 r8 | | | |
| lr1e-4 r16 | | | |
| lr2e-4 r8 | | | |
| lr2e-4 r16 | | | |

## 5. GPU / CUDA (itens 5 e 7)
RTX 3060 Ti = arquitetura Ampere (compute 8.6), suporta bf16.
1. Driver NVIDIA atualizado (`nvidia-smi` deve funcionar).
2. Instale o **PyTorch já com CUDA** (o toolkit CUDA completo não é obrigatório para o PyTorch, mas se o professor exigir, instale o CUDA Toolkit 12.1):
   `pip install torch --index-url https://download.pytorch.org/whl/cu121`
3. `pip install -r requirements.txt`
4. `python src/check_gpu.py` → deve mostrar `CUDA disponível: True` e `NVIDIA GeForce RTX 3060 Ti`.
5. **Print:** extensões VSCode **GPU Monitor** e **GPU Environment** abertas + terminal com `check_gpu.py` (e `nvidia-smi` durante o treino mostrando uso de VRAM). Salve em `docs/print_gpu.png`.

## 6. Como rodar
```bash
python -m venv .venv && .venv\Scripts\activate        # Windows
python src/check_gpu.py
python src/make_dataset.py
python src/train.py --out outputs/run1                # ~10-20 min na 3060 Ti
python src/evaluate.py --adapter outputs/run1
python src/app.py --adapter outputs/run1              # frontend em http://127.0.0.1:7860
```
Se faltar memória: reduza `per_device_train_batch_size` para 2 em `train.py`.

## 7. Estrutura
`src/` código · `data/` dataset · `results/` métricas e amostras · `docs/` prints · `outputs/` adaptadores (no .gitignore)

## 8. Checklist de entrega
- [ ] Repositório no GitHub com README preenchido (tabela de resultados)
- [ ] `docs/print_gpu.png` (VSCode + GPU funcionando)
- [ ] Print do frontend respondendo
- [ ] Resultados de `results/` commitados
- [ ] Apresentação: modelo, problema, parâmetros, resultados, limitações
