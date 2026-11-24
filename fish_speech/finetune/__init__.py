"""Drama 3 single-speaker fine-tuning (LoRA or partial full fine-tuning).

Modules:
    sequence  training sequences in the exact Drama 3 prompt format, with loss masks
    data      prepared-manifest dataset, task mixture and collator
    model     teacher-forced training wrapper around FishOmniModel (losses, checkpointing)
    peft_utils  LoRA targets, adapter loading and merging

Entry points live in tools/finetune/ (prepare_data, train, infer, export_merged,
evaluate, synthesize_replay). See docs/en/finetune.md.
"""
