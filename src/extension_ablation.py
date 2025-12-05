import json
import re
import networkx as nx


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


def run_ablation(entities, relations):
    """
    Compares the original KG to the KG where entity strings are lowercased
    """

    from kg_build import build_graphs
    G_gt, G_pred = build_graphs(entities, relations)

    ents_norm = []
    for e in entities:
        try:
            cleaned = _clean_json(e)
            ent_list = json.loads(cleaned)
            # Handle dict format
            normalized = []
            for item in ent_list:
                if isinstance(item, dict):
                    item_copy = item.copy()
                    if "entity" in item_copy:
                        item_copy["entity"] = item_copy["entity"].lower()
                    if "text" in item_copy:
                        item_copy["text"] = item_copy["text"].lower()
                    normalized.append(item_copy)
                elif isinstance(item, str):
                    normalized.append(item.lower())
            ents_norm.append(json.dumps(normalized))
        except:
            ents_norm.append("[]")

    rels_norm = []
    for r in relations:
        try:
            cleaned = _clean_json(r)
            rlist = json.loads(cleaned)
            for row in rlist:
                row["source"] = row["source"].lower()
                row["target"] = row["target"].lower()
            rels_norm.append(json.dumps(rlist))
        except:
            rels_norm.append("[]")

    G_gt2, G_pred2 = build_graphs(ents_norm, rels_norm)

    return {
        "original_nodes": len(G_pred.nodes()),
        "normalized_nodes": len(G_pred2.nodes()),
        "original_edges": len(G_pred.edges()),
        "normalized_edges": len(G_pred2.edges()),
    }