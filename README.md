# text2sql-lora-finetuning

In this project, our group fine tunes the model flan-t5-base using LoRA as the adapter with the purpose of improving the ability of LLM to translate natural texts to sql queries. Three LLMs, flan-t5-base, flan-t5-xl,and flan-t5-base with LoRA, are trained and evaluated on spider database from https://github.com/taoyds/spider. Thereafter, 5 metrics: execution accuracy, exact matching accuracy, partial matching accuracy, partial matching recall, and partial matching F1 for each clause are measured to compare the performance across the three models.

During the training process, each training example includes a natural-language question, the database schema, and the corresponding SQL query. The model trains on small LoRA adapter matrices inserted into attention and feed-forward projections (q,k,v,o,wi_0,wi_1,w0). And then the Hugging Face, Seq2SeqTrainer, optimizes the model cross-entropy loss with label smoothing (0.1) over three epochs, evaluated by exact-match accuracy each epoch. The best checkpoint is saved and applied to the model.

```
#Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Folder with train/val/tables json files")
    ap.add_argument("--train_json", default="train_spider.json")
    ap.add_argument("--val_json",   default="dev.json")
    ap.add_argument("--tables_json", default="tables.json")
    ap.add_argument("--model_name", default="google/flan-t5-base")
    ap.add_argument("--output_dir", default="nl2sql-lora-out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_src_len", type=int, default=512)
    ap.add_argument("--max_tgt_len", type=int, default=160)
    args = ap.parse_args()

# Model + LoRA (broader targets), no 8-bit to keep it simple/stable
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q","k","v","o","wi_0","wi_1","wo"],  # attention + FFN
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Trainer
    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,

        # Stability while debugging
        fp16=False, bf16=False,
        label_smoothing_factor=0.1,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        tokenizer=tok
    )

    trainer.train()
```

Across the three evaluations, the LoRA-finetuned model shows great advantage over Flan-T5-base and Flan-T5-XL in both execution accuracy (0.299 to 0.007 and 0.115) and exact match accuracy (0.292 to 0.007 to 0.103). Therefore, both the queried results and the sql queries from the finetuned model more accurately aligned with the expected output. And for the clause-level F1 score, LoRA achieves 0.4-0.6 across SELECT, WHERE, GROUP, and ORDER clauses, while Flan-T5-XL's score hover near 0.1-0.3, and Flan-T5-base is almost zero. LoRA's F1 score also degrades smoothly as complexity rises, whereas the other two models collapse on medium-to-hard SQLs. In short, LoRA fine-tuning greatly strengthen SQL generation, outperforming the base and XL models on nearly every metrics.

The evaluation results are in LoRA_finetuned.png, Flan-T5-base.png, and Flan-T5-XL.png.
