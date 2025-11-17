import os, json, argparse, re, csv
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
import torch

# ---------- utils

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_sql(sql: str) -> str:
    s = re.sub(r"\\s+", " ", (sql or "")).strip().lower()
    s = re.sub(r"(<=|>=|=|<|>)", lambda m: f" {m.group(0)} ", s)
    return re.sub(r"\\s+", " ", s).strip().rstrip(";")

# ---------- schema formatting (compact, capped)

def build_schema_texts(tables_json_path: str, keep_db_ids: set=None, cap_cols=6, cap_fks=6) -> Dict[str, str]:
    """
    Returns {db_id: compact_schema_text} with capped columns & FKs to keep prompts short.
    """
    tables = load_json(tables_json_path)
    out = {}
    for db in tables:
        db_id = db["db_id"]
        if keep_db_ids and db_id not in keep_db_ids:
            continue

        names = db["table_names_original"]          # list[str]
        cols  = db["column_names_original"]         # list[[t_idx, col_name]]
        pks   = set(db["primary_keys"])
        fks   = db["foreign_keys"]

        per_table = {i: [] for i in range(len(names))}
        for idx, (t_idx, c_name) in enumerate(cols):
            if t_idx == -1:
                continue
            tag = " PK" if idx in pks else ""
            per_table[t_idx].append(f"{c_name}{tag}")

        fk_lines = []
        for child, parent in fks:
            ct, cc = cols[child]
            pt, pc = cols[parent]
            if ct == -1 or pt == -1:
                continue
            fk_lines.append(f"{names[ct]}.{cc}->{names[pt]}.{pc}")

        lines = [f"DB: {db_id}", "Tables:"]
        for i, t in enumerate(names):
            c = ", ".join(per_table[i][:cap_cols])
            if len(per_table[i]) > cap_cols:
                c += ", ..."
            lines.append(f"- {t}({c})")

        if fk_lines:
            if len(fk_lines) > cap_fks:
                fk_show = "; ".join(fk_lines[:cap_fks]) + "; ..."
            else:
                fk_show = "; ".join(fk_lines)
            lines.append("FKs: " + fk_show)

        out[db_id] = "\\n".join(lines)
    return out

# ---------- dataset construction (whitelist fields)

def make_hf_datasets(root: str, train_json: str, val_json: str, schema_texts: Dict[str,str]):
    train_raw = load_json(os.path.join(root, train_json))
    val_raw   = load_json(os.path.join(root, val_json))

    def to_record(x):
        q = str(x.get("question","")).strip()
        sql = str(x.get("query","")).strip()
        db = str(x.get("db_id","")).strip()
        schema = schema_texts.get(db, f"DB: {db}\\nTables: (missing)")
        return {
            "input_text": f"translate to sql: {q}\\n{schema}",
            "labels": sql,
            "db_id": db,
            "question": q,
        }

    train_ds = Dataset.from_list([to_record(z) for z in train_raw])
    val_ds   = Dataset.from_list([to_record(z) for z in val_raw])
    return train_ds, val_ds

def tokenize_datasets(train_ds: Dataset, val_ds: Dataset, tok, max_src=512, max_tgt=160):
    def _prep(batch):
        mi = tok(batch["input_text"], max_length=max_src, truncation=True)
        with tok.as_target_tokenizer():
            labels = tok(batch["labels"], max_length=max_tgt, truncation=True)
        mi["labels"] = labels["input_ids"]
        return mi
    train_tok = train_ds.map(_prep, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(_prep,   batched=True, remove_columns=val_ds.column_names)
    return train_tok, val_tok

# ---------- EM evaluation with truncation + safe decoding

def exact_match_eval(model, tok, val_ds, out_csv=None, max_gen_len=160, num_beams=4, max_src_len=512):
    device = next(model.parameters()).device
    preds, refs, rows = [], [], []

    for ex in val_ds:
        prompt = ex["input_text"]
        gold   = ex["labels"]
        db_id  = ex.get("db_id", "")
        schema = ex["input_text"].split("\\n",1)[1] if "\\n" in ex["input_text"] else ""

        enc = tok(prompt, return_tensors="pt", max_length=max_src_len, truncation=True).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_gen_len,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
                early_stopping=True,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
            )
        pred = tok.decode(out_ids[0], skip_special_tokens=True)

        preds.append(normalize_sql(pred))
        refs.append(normalize_sql(gold))
        rows.append({"db_id": db_id, "question": ex.get("question",""), "pred": pred, "gold": gold, "schema": schema})

    em = (sum(p == r for p, r in zip(preds, refs)) / len(refs)) if refs else 0.0

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["db_id","question","pred","gold","schema"])
            w.writeheader()
            w.writerows(rows)

    return em

# ---------- main

def main():
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

    train_path = os.path.join(args.data_root, args.train_json)
    val_path   = os.path.join(args.data_root, args.val_json)
    tables_path= os.path.join(args.data_root, args.tables_json)

    # Build compact schema texts for only DBs we need
    train_items = load_json(train_path)
    val_items   = load_json(val_path)
    keep_db_ids = set([x["db_id"] for x in train_items + val_items])
    schema_texts = build_schema_texts(tables_path, keep_db_ids=keep_db_ids, cap_cols=6, cap_fks=6)

    # Datasets & tokenizer
    train_ds, val_ds = make_hf_datasets(args.data_root, args.train_json, args.val_json, schema_texts)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    train_tok, val_tok = tokenize_datasets(train_ds, val_ds, tok, args.max_src_len, args.max_tgt_len)

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

    # Save LoRA adapters + tokenizer
    adapters_dir = os.path.join(args.output_dir, "lora_adapters")
    os.makedirs(adapters_dir, exist_ok=True)
    model.save_pretrained(adapters_dir)
    tok.save_pretrained(adapters_dir)

    # Quick EM + judge CSV
    em = exact_match_eval(
        model, tok, val_ds,
        out_csv=os.path.join(args.output_dir, "judge_input.csv"),
        max_gen_len=args.max_tgt_len,
        num_beams=4,
        max_src_len=args.max_src_len
    )
    print(f"[VAL] Exact Match: {em:.4f}")
    print(f"Saved adapters to: {adapters_dir}")
    print(f"Saved judge CSV to: {os.path.join(args.output_dir, 'judge_input.csv')}")

if __name__ == "__main__":
    main()

