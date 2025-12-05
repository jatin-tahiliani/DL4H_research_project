import os
import time
import json
import re
from openai import OpenAI


# Initialize OpenAI client using env variables
client = OpenAI(
   api_key=os.getenv("OPENAI_API_KEY"),
   organization=os.getenv("OPENAI_ORG_ID"),
   project=os.getenv("OPENAI_PROJECT")
)


SYSTEM_PROMPT = """
You are an expert biomedical NER model.
Extract clinical entities from radiology reports.


Use these labels exactly:
- Anatomy
- Observation
- Diagnosis
- Device


Return ONLY a JSON array of objects.
Each object must have:
{
 "text": "<entity text>",
 "label": "<Anatomy|Observation|Diagnosis|Device>"
}
"""




def clean_json(text):
   """
   Remove ```json fences, backticks, and extract
   the first valid JSON array from the string.
   """
   if text is None:
       return "[]"


   # Remove ```json or ``` blocks
   text = re.sub(r"```[a-zA-Z]*", "", text)
   text = text.replace("```", "").strip()


   # Attempt to extract JSON array using regex
   match = re.search(r"\[.*\]", text, re.DOTALL)
   if match:
       cleaned = match.group(0).strip()
       return cleaned


   # If no array found, return empty
   return "[]"


# ---------------------------------------------------
# NER extraction function
# ---------------------------------------------------
def extract_entities(df):
   """
   Takes a dataframe with a 'report' column.
   Returns a list of cleaned JSON strings,
   one for each radiology report.
   """


   all_entities = []


   for i, report in enumerate(df["report"]):
       print(f"Running NER {i+1}/{len(df)} ...")


       # Build prompt
       user_prompt = (
           "Extract all clinical entities from this radiology report:\n\n"
           f"{report}\n\n"
           "Return ONLY a JSON array. No explanations."
       )


       # Try/retry loop (handles rate limits)
       for attempt in range(3):
           try:
               response = client.chat.completions.create(
                   model="gpt-4o-mini",
                   messages=[
                       {"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": user_prompt}
                   ],
                   temperature=0
               )


               raw = response.choices[0].message.content
               cleaned = clean_json(raw)


               # Validate JSON
               try:
                   json.loads(cleaned)
               except Exception:
                   print("Warning: model returned invalid JSON, forcing [].")
                   cleaned = "[]"


               all_entities.append(cleaned)
               break  # success → exit retry loop


           except Exception as e:
               print(f"NER error on report {i+1}: {e}")
               time.sleep(1)


               if attempt == 2:
                   print("Giving up after 3 attempts.")
                   all_entities.append("[]")


   return all_entities