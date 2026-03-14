# ================== GPT VERSION (OpenAI Responses API) ==================
# pip install --upgrade openai

import os
import re
import json
import time
import glob

# ========== API SETUP ==========
api_key = "ENTER API KEY"

USE_RESPONSES_API = False
client = None

try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    USE_RESPONSES_API = True
except Exception:
    import openai as openai_legacy
    openai_legacy.api_key = api_key
    USE_RESPONSES_API = False


OPENAI_MODEL_ID = "GPT Model"


# ========== FILE SETUP ==========
input_folder = r"Input Path"
output_folder = os.path.join(input_folder, "GPT", "Validation_Results")

os.makedirs(output_folder, exist_ok=True)

target_files = sorted(
    glob.glob(os.path.join(input_folder, "*.json"))
    + glob.glob(os.path.join(input_folder, "*.jsonl"))
)


# ========== SETTINGS ==========
HISTORY_WINDOW_TURNS = 12
MAX_RETRIES = 3
RETRY_DELAY = 2


# ========== PROMPTS ==========

def get_validation_prompt(history_text):

    return f"""
You are a Clinical Dialogue Evaluator. Your task is to perform a binary classification of the patient's most recent turn in a doctor–patient conversation.

Decision Rules:
A response is correct if it is:
1. Relevant to the doctor's immediately preceding question or context.
2. Medically appropriate in the given context.
3. Logically consistent with all previous dialogue.

A response is incorrect if it meets any of the following:
Contradiction – conflicts with earlier responses
Incomplete – partially answers the question
Irrelevant – unrelated to the question or context
Too Much Information – excessive unnecessary detail
Vague – unclear or ambiguous language

Examples of Correct responses:
Doctor: "When did the pain start?"
Patient: "It began last night."

Doctor: "Do you smoke?"
Patient: "No, I don't."

Doctor: "Does it hurt more when you lie down?"
Patient: "Yes, lying down makes it worse."

Examples of Incorrect responses:
Doctor: "When did the pain start?"
Patient: "Uh maybe last night maybe the day before not sure."

Doctor: "Do you smoke?"
Patient: "Yes and no."

Doctor: "Where exactly is the pain located?"
Patient: "... side."

Doctor: "Do you feel short of breath?"
Patient: "Well I also run every day and last month it was hot..."

Doctor: "Any chest tightness?"
Patient: "My neighbor bought a dog last week."

Conversation History
{history_text}

Output format (exactly one line):
Accuracy: Correct
or
Accuracy: Incorrect
"""


def get_error_prompt(history_text):

    return f"""
You are a Clinical Dialogue Error Type Classifier.

This prompt is invoked only after the binary classifier has judged the most recent patient turn as Incorrect.

Task:
Select the one primary error category that best describes why the latest patient turn is incorrect.

Error Categories:

Contradiction – Conflicts with prior dialogue or with itself.

Incomplete – Fails to provide the key information requested.

Irrelevant – Entirely unrelated to the doctor's question.

Too Much Information – Adds excessive or tangential details beyond what was needed.

Vague – Ambiguous, uncertain, or lacking specificity.

Use only the provided conversation history.

Conversation History
{history_text}

Output format (exactly one line):
Error Type: <one category from the list above ONLY>
"""


# ========== RESPONSE PARSERS ==========

def extract_response_text(resp):

    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()

    try:
        parts = []
        for item in resp.output:
            for content in item.content:
                if hasattr(content, "text"):
                    parts.append(content.text)

        return "".join(parts).strip()

    except:
        return str(resp).strip()


def extract_chat_text(resp):

    try:
        return resp["choices"][0]["message"]["content"].strip()
    except:
        return str(resp)


# ========== MODEL CALLER ==========

def call_model(prompt):

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            if USE_RESPONSES_API:

                resp = client.responses.create(
                    model=OPENAI_MODEL_ID,
                    input=prompt,
                    temperature=0,
                    max_output_tokens=200
                )

                return extract_response_text(resp)

            else:

                resp = openai_legacy.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200
                )

                return extract_chat_text(resp)

        except Exception as e:

            last_error = e
            time.sleep(RETRY_DELAY * (2 ** attempt))

    raise RuntimeError(f"Model call failed: {last_error}")


# ========== FILE LOADER ==========

def load_json_file(file_path):

    with open(file_path, "r", encoding="utf-8") as f:

        text = f.read().strip()

        try:
            data = json.loads(text)

            if isinstance(data, list):
                return data

        except:
            pass

        rows = []

        for line in text.splitlines():

            try:
                rows.append(json.loads(line))
            except:
                continue

        return rows


# ========== ROLE CHECKERS ==========

def is_doctor(role):

    return str(role).lower().strip() == "doctor"


def is_patient(role):

    r = str(role).lower().strip()

    return r in ["patient", "error_patient"]


# ========== MAIN PROCESSOR ==========

def process_file(file_path):

    rows = load_json_file(file_path)

    rows.sort(key=lambda x: (x.get("turn", 0)))

    history = []

    results = []

    last_doctor = ""
    last_patient = ""

    for item in rows:

        role = item.get("role")

        msg = (item.get("message") or "").strip()

        if not msg:
            continue

        if is_doctor(role):

            last_doctor = msg
            history.append(("Doctor", msg))


        elif is_patient(role):

            last_patient = msg
            history.append(("Patient", msg))

            recent = history[-HISTORY_WINDOW_TURNS:]

            history_text = "\n".join(f"{r}: {m}" for r, m in recent)

            validation_prompt = get_validation_prompt(history_text)

            validation_output = call_model(validation_prompt)

            text = validation_output.lower()

            if "accuracy: correct" in text:
                accuracy = "Correct"
            else:
                accuracy = "Incorrect"


            error_type = ""

            if accuracy == "Incorrect":

                error_prompt = get_error_prompt(history_text)

                error_output = call_model(error_prompt)

                match = re.search(r"Error Type:\s*(.*)", error_output, re.I)

                if match:
                    error_type = match.group(1).strip()


            print("\n--------------------------------------------------")
            print("Doctor:", last_doctor)
            print("Patient:", last_patient)
            print("Accuracy:", accuracy)
            print("Error Type:", error_type if error_type else "N/A")
            print("--------------------------------------------------")


            results.append({
                "Doctor": last_doctor,
                "Patient": last_patient,
                "Accuracy": accuracy,
                "Error Type": error_type
            })


    base = os.path.splitext(os.path.basename(file_path))[0]

    output_path = os.path.join(output_folder, f"{base}_validated_GPT.json")

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved:", output_path)



# ========== RUN ==========

if __name__ == "__main__":

    if not target_files:
        print("No JSON files found")

    for f in target_files:

        print("\nProcessing", os.path.basename(f))

        try:
            process_file(f)

        except Exception as e:

            print("Error processing file:", e)