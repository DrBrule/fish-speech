# Fine-tuning Drama 3 for a single default voice

This recipe makes **your speaker the voice Drama 3 uses when no reference is given**, while keeping what the base model already does: about 80 languages, inline and scoped tags, style prefixes, director's notes, multi-turn revision, multi-speaker scenes, and cloning other voices from references.

It adds training on top of the inference stack that ships in this repository. The training forward pass runs the same backbone and codebook-MTP modules as inference (`fish_speech/models/text2semantic/fish_omni`). Prompts come from the same builder (`build_drama3_prompt`), so what the model learns is exactly what the CLI, the API server and SGLang-Omni send it. Hugging Face pieces do the rest: `transformers.Trainer` (training loop, accelerate/DDP, checkpoints, logging), `peft` (LoRA), and `transformers` pipelines for ASR and speaker embeddings in data preparation and evaluation.

!!! warning "License and consent"
    Fine-tuned weights are Derivative Works under the Fish Audio Community License. Only train on a voice you have the right and the consent to use. The license also restricts using model outputs to train *other* speech models; `synthesize_replay.py` trains Drama 3 on its own outputs, so check that your use is covered.

## How it works

Drama 3 is trained as a conversation (technical report §3, Appendix A/B). The system turn holds the task instruction and any `[Reference N]` voice clips, the user turn holds the text, and the assistant turn holds the audio:

```
<|im_start|>system
convert the provided text to speech<|im_end|>
<|im_start|>user
<|speaker:0|> [sighs] Fine. If we miss it, it's on you.<|im_end|>
<|im_start|>assistant
<|voice|>{{audio frames}}<|im_end|>
```

With no reference, the system turn is just `convert the provided text to speech`, and the base model picks a voice on its own. Fine-tuning supervises that exact prompt with your recordings, so "no reference" comes to mean "your voice". The loss is the paper's (Eq. 8–9): `L = L_text + 0.5·L_sem + 0.5·L_cb`. It covers the semantic code from the backbone, the nine acoustic codes from the codebook-MTP decoder, and the closing `<|im_end|>`. Reference audio and the prompt are context only.

Each training example draws a task (`fish_speech/finetune/data.py`):

| Task | Prompt | Why |
| --- | --- | --- |
| `default` (55%) | no reference | makes your voice the default |
| `history` (20%) | no reference + 1–2 earlier turns of the same recording | long texts are generated in chunks with earlier chunks as history; keeps the voice stable across chunks |
| `clone` (25%) | 1–2 `[Reference N]` clips of your voice | keeps the cloning path intact and consistent with the default voice |
| replay rows | always `clone`, references from the same other speaker | optional; keeps cloning other voices and other languages working (pairs with `--kl_weight`) |

The capabilities are kept in four ways:

1. **Direction stays in the text.** Transcripts keep inline tags (`[laughs]`, `<whisper>…</whisper>`). A `style` field, if given, is written as a style prefix (`[warm, smiling] line`) or a director's note (`Direction: … \nLine: …`), in the templates the model was trained on.
2. **Self-distilled replay** (`synthesize_replay.py`). The base model clones your voice on multilingual and directed lines. Takes that pass ASR and speaker-similarity checks become extra *default-voice* rows. Your voice then becomes the default for the inputs your real data does not cover, such as other languages, whispers and laughter, so it does not only replace the neutral voice in your own language.
3. **Small, targeted updates.** LoRA on the dense backbone projections (attention, Gated DeltaNet, shared expert) and on the codebook-MTP tower. The tower carries most of the timbre. The 256 routed experts per layer, the router, the embeddings and the `lm_head` stay frozen, so the language and text knowledge is untouched.
4. **Optional KL to the base model** (`--kl_weight`) on replay rows of other voices. It holds the semantic-token distribution where the base model's behaviour is what you want.

## Requirements

- An NVIDIA GPU setup that runs Drama 3 inference. For LoRA with FP8 experts, one 80 GB SM89+ GPU (H100, H200, B200) is enough: weights take about 42 GB, plus activations. Lower `--max_rows` on smaller cards.
- On A100 or other pre-SM89 GPUs the FP8 experts are dequantized to bf16 (about 77 GB of weights). Spread one copy over two GPUs with `--devices cuda:0,cuda:1`.
- Extra Python packages: `pip install -e ".[finetune]"` (adds `peft`, `accelerate`).
- Checkpoint: `hf download fishaudio/drama-3 --local-dir checkpoints/drama-3`.

## 1. Prepare the data

**What to record.** Aim for 30 minutes or more of clean speech from one speaker. Several hours is better. Cut it into 2–20 s clips at 44.1 kHz or higher, with accurate transcripts. Write non-verbal events and delivery into the transcript where they happen (`[laughs]`, `[sighs]`, `[pause]`, `<whisper>…</whisper>`). Add a `style` for clips with a clear delivery ("excited", "tired and dry"). Varied delivery in the data is the best way to keep the default voice directable.

Either put audio next to transcripts (`clip.wav` + `clip.lab`), one folder per recording session in order:

```bash
python tools/finetune/prepare_data.py \
    --input data/myvoice --output-dir data/prepared \
    --speaker myvoice --language en --consecutive
```

or give a JSONL manifest (`audio`, `text`, and optional `language`, `style`, `group`, `order`):

```bash
python tools/finetune/prepare_data.py --input data/myvoice.jsonl --output-dir data/prepared --speaker myvoice
```

`--consecutive` marks the files of a folder as consecutive segments of one recording, which enables the `history` task. `--transcribe` fills in missing transcripts with Whisper. Clips are encoded once with the Drama 3 codec (10 codebooks, about 21.5 frames/s) into `data/prepared/codes/*.npy`, and listed in `data/prepared/manifest.jsonl`.

**Self-distilled replay (recommended):**

```bash
python tools/finetune/synthesize_replay.py \
    --checkpoint-path checkpoints/drama-3 \
    --target-manifest data/prepared/manifest.jsonl \
    --output-dir data/prepared --takes 3
```

This uses the lines in `tools/finetune/replay_prompts.jsonl`: 40 lines in 17 languages with tags and styles. Add lines in the languages and styles you care about. Aim for replay to be 10–20% of your rows. Takes are kept only when Whisper's error rate is at most `--max-wer`, and the best of `--takes` by speaker similarity wins.

**Other voices (optional):** add clips of other speakers (two or more clips each) as replay rows, and train with `--kl_weight 0.05`:

```bash
python tools/finetune/prepare_data.py --input data/others.jsonl --output-dir data/prepared \
    --role replay --append
```

## 2. Train

```bash
python tools/finetune/train.py tools/finetune/configs/lora_single_gpu.yaml
```

Or set the flags directly; every `transformers.TrainingArguments` flag works:

```bash
python tools/finetune/train.py \
    --checkpoint_path checkpoints/drama-3 \
    --train_manifest data/prepared/manifest.jsonl \
    --output_dir results/drama3-myvoice \
    --mode lora --lora_preset backbone --lora_r 32 --lora_alpha 32 --tower lora \
    --learning_rate 1e-4 --lr_scheduler_type cosine --warmup_steps 0.03 \
    --num_train_epochs 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --bf16 --logging_steps 10 --save_steps 500 --report_to none
```

Multi-GPU data parallel (one full copy per GPU): `accelerate launch --num_processes 8 tools/finetune/train.py …`. One copy spread over smaller GPUs (single process): `--devices cuda:0,cuda:1,cuda:2`. The last device also holds the tower.

Logs show `loss_text`, `loss_semantic`, `loss_codebook` (and `loss_kl`, `loss_router_aux` when enabled). `--resume_from_checkpoint` resumes adapters, full weights and optimizer state.

### Modes

| Flags | Trains | Notes |
| --- | --- | --- |
| `--mode lora --tower lora` (default) | LoRA on backbone + tower | smallest change, best at keeping abilities |
| `--mode lora --tower full --tower_learning_rate 2e-5` | LoRA on backbone, full codebook-MTP tower (≈1.1B) | closer timbre on larger datasets (hours) |
| `--mode full --full_scopes tower,backbone_dense` | every dense weight except routed experts, routers, embeddings, `lm_head` | ≈2.6B trainable; roughly 15–30 GB of gradients and optimizer state on top of the weights (`--optim adamw_8bit` helps); no `--kl_weight` |

The routed experts are not trained in any mode. They hold about 32B of the 35B parameters, they are FP8 in the release, and one voice does not need them. During training they still pass gradients to the layers below: FP8 experts are dequantized per expert in the forward pass.

### Recipe

- **Learning rate:** 1e-4 for LoRA (r 32). Use 5e-5 if multilingual or tag outputs degrade in evaluation.
- **Steps:** 2–4 epochs over 1–5 h of audio. Watch `loss_semantic` on a held-out `--eval_manifest`, and stop when it flattens.
- **Batch:** keep the effective batch at 16 or more sequences (`gradient_accumulation_steps`).
- **Mixture:** raise `--clone_prob` if cloning other voices from references gets worse. Raise `--history_prob` if long texts drift between chunks.

## 3. Use it

Merged in memory with the base checkpoint:

```bash
python tools/finetune/infer.py --checkpoint-path checkpoints/drama-3 \
    --finetune results/drama3-myvoice --output out.wav \
    --text "[with bright eager excitement] We won! <whisper>Don't tell anyone yet.</whisper>"
```

Or export a stand-alone checkpoint (release layout, FP8 experts unchanged, only the changed tensors rewritten) and use the stock tools:

```bash
python tools/finetune/export_merged.py --checkpoint-path checkpoints/drama-3 \
    --finetune results/drama3-myvoice --output checkpoints/drama-3-myvoice
python tools/api_server.py --llama-checkpoint-path checkpoints/drama-3-myvoice --device cuda:0
```

A request with no reference now speaks in your voice. References, `<|speaker:N|>` tags and every control in the [syntax reference](inference.md) still work. Multi-speaker scenes cast the other speakers from their references as before.

## 4. Evaluate

```bash
python tools/finetune/evaluate.py --checkpoint-path checkpoints/drama-3 \
    --finetune results/drama3-myvoice --target-manifest data/prepared/manifest.jsonl \
    --output-dir results/eval
```

This generates every prompt in `tools/finetune/eval_prompts.jsonl` (plain, point tags, scoped tags, style prefixes, director's notes, 12 languages) with no reference, first from the base model and then from the fine-tune. It reports:

- `sim`: speaker similarity to your recordings (WavLM speaker verification). It should rise on every kind of prompt, multilingual and tagged included.
- `wer`: Whisper error rate against the text with tags removed (CER for zh/ja/ko/th/yue). It should stay close to the base model. A jump on the multilingual or tagged rows means those abilities are being lost.

Delivery is not scored automatically, so listen to the `*_tag`, `style_prefix` and `director_note` files in `results/eval/finetuned/`.

## Code map

| Path | Purpose |
| --- | --- |
| `fish_speech/finetune/sequence.py` | training sequences from `build_drama3_prompt` + supervised assistant audio |
| `fish_speech/finetune/data.py` | manifest, task mixture, collator |
| `fish_speech/finetune/model.py` | teacher-forced backbone + tower losses, gradient checkpointing, KL, router aux |
| `fish_speech/finetune/peft_utils.py` | LoRA targets, full-weight scopes, save / load / merge |
| `tools/finetune/*.py` | prepare_data, synthesize_replay, train, infer, export_merged, evaluate |
| `tests/test_finetune.py` | tiny-model checks: prompt parity, cached vs teacher-forced tower, LoRA step, save/merge round trip |
