import json
import re
import networkx as nx


def _clean_json(text):
    """Strip markdown code blocks and extract JSON."""
    if text is None:
        return "[]"
    # Remove ```json or ``` blocks
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "").strip()
    # Extract JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return "[]"


def _safe_json_load(x):
    try:
        cleaned = _clean_json(x)
        return json.loads(cleaned)
    except:
        return []


def build_graphs(entities, relations):
    """
    Build two graphs which is the ground truth with only entities and a predicted graph
    """
    G_gt = nx.Graph()
    G_pred = nx.Graph()

    for ent_list, rel_list in zip(entities, relations):
        ents = _safe_json_load(ent_list)
        rels = _safe_json_load(rel_list)

        for e in ents:
            # Handle dict format: {"entity": "...", "category": "..."}
            # or {"text": "...", "label": "..."}
            if isinstance(e, dict):
                node_name = e.get("entity") or e.get("text") or str(e)
                G_gt.add_node(node_name)
                G_pred.add_node(node_name)
            elif isinstance(e, str):
                G_gt.add_node(e)
                G_pred.add_node(e)

        for triplet in rels:
            try:
                s = triplet["source"]
                t = triplet["target"]
                r = triplet["relation"]
                G_pred.add_edge(s, t, relation=r)
            except:
                continue

    return G_gt, G_pred