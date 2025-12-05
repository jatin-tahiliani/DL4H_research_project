from openai import OpenAI
import json
import re
import time
import os


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT")
)


RE_SYSTEM_PROMPT = """
You are an expert in biomedical relation extraction.
Given a radiology report and its extracted entities,
identify relations in this format:
[
  {"source": "...", "target": "...", "relation": "..."}
]
Relations to look for: located_in, associated_with, indicates, caused_by, part_of
"""


def clean_json(text):
    """Strip markdown code blocks and extract JSON."""
    if text is None:
        return "[]"
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return "[]"


def extract_relations(df, entities):
    """
    Use GPT-4o-mini to extract relations between entities.
    """
    all_relations = []

    for i, (report, ent) in enumerate(zip(df["report"], entities)):
        print(f"Running RE {i+1}/{len(df)} ...")

        prompt = f"""
Report:
{report}

Entities:
{ent}

Extract relations in JSON format only.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": RE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0
            )

            rel = response.choices[0].message.content
            cleaned = clean_json(rel)

            # Validate JSON
            try:
                json.loads(cleaned)
            except:
                cleaned = "[]"

            all_relations.append(cleaned)

        except Exception as e:
            print("RE error:", e)
            all_relations.append("[]")
            time.sleep(1)

    return all_relations