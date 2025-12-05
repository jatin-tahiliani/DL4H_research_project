import matplotlib.pyplot as plt
import pandas as pd
import json
import re


def _clean_json(text):
    """Strip markdown code blocks and extract JSON."""
    if text is None:
        return "[]"
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return "[]"


def plot_report_length_hist(df, save_path):
    if df.empty:
        print("Warning: reports_df is empty. Skipping histogram.")
        return

    # Calculate lengths from report column
    lengths = df["report"].str.len()

    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=20, edgecolor='black')
    plt.title("Distribution of Report Lengths")
    plt.xlabel("Report Length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_disorder_freq(entities, save_path):
    flat = []
    for ent_json in entities:
        if ent_json and ent_json != "[]":
            try:
                cleaned = _clean_json(ent_json)
                parsed = json.loads(cleaned)
                for e in parsed:
                    if isinstance(e, dict):
                        # Handle both "entity" and "text" keys
                        text = e.get("entity") or e.get("text") or ""
                        if text:
                            flat.append(text)
                    elif isinstance(e, str):
                        flat.append(e)
            except:
                pass

    if len(flat) == 0:
        print("Warning: No disorders/entities found. Skipping disorder frequency plot.")
        return

    freq = pd.Series(flat).value_counts().head(10)

    plt.figure(figsize=(10, 6))
    freq.plot(kind="bar")
    plt.title("Top Entities")
    plt.xlabel("Entity")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metric_bar(data_dict, save_path):
    if not data_dict:
        print("Warning: Empty metric dict. Skipping metric plot.")
        return

    names = list(data_dict.keys())
    values = list(data_dict.values())

    plt.figure(figsize=(6, 4))
    plt.bar(names, values)
    plt.ylabel("Score")
    plt.title("KG Evaluation Metrics")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_ablation(df_or_dict, save_path):
    if isinstance(df_or_dict, dict):
        df = pd.DataFrame([df_or_dict])
    else:
        df = df_or_dict

    df.plot(kind="bar", figsize=(6, 4))
    plt.title("Ablation: Node/Edge Counts Before vs After Normalization")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()