#!/usr/bin/env python3
"""
Clinical Dialogue Validator
Two-stage error detection and classification for goal-oriented medical conversations
Binary Classification: Correctness → Multi-class: Error Type
Anonymous implementation - no hardcoded paths or credentials
"""

import os
import re
import json
import time
import glob
import argparse
import sys
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class DialogueValidator:
    """Two-stage validator: Binary classification + Multi-class error classification"""

    # ========== PROMPTS ==========
    BINARY_VALIDATION_PROMPT = """You are a Clinical Dialogue Evaluator. Your task is to perform a binary classification of the patient's most recent turn in a doctor–patient conversation.

Decision Rules:
A response is correct if it is:
1. Relevant to the doctor's immediately preceding question or context.
2. Medically appropriate in the given context.
3. Logically consistent with all previous dialogue.

A response is incorrect if it meets any of the following:
- Contradiction: when it states information that conflicts with earlier responses
- Incomplete: when it provides only part of the required answer or leaves the question unanswered
- Irrelevant: when it includes content unrelated to the doctor's question or medical context
- Too Much Information: when it gives excessive or unnecessary detail beyond what was asked
- Vague: when it uses unclear, ambiguous, or non-specific language that does not directly answer the question

Examples of Correct responses:
Doctor: "When did the pain start?"
Patient: "It began last night."

Doctor: "Do you smoke?"
Patient: "No, I don't."

Doctor: "Does it hurt more when you lie down?"
Patient: "Yes, lying down makes it worse."

Examples of Incorrect responses:
Doctor: "When did the pain start?"
Patient: "Uh... maybe last night, maybe the day before, not sure." (Vague)

Doctor: "Do you smoke?"
Patient: "Yes and no." (Contradiction)

Doctor: "Where exactly is the pain located?"
Patient: "... side." (Incomplete)

Doctor: "Do you feel short of breath?"
Patient: "Well, I also run every other day, and last month it was hot, and I was drinking less water, and I changed my sleeping schedule, and sometimes I stay up late watching movies, so I get really tired on top of everything." (Too Much Information)

Doctor: "Any chest tightness?"
Patient: "My neighbor bought a dog last week." (Irrelevant)

Conversation History:
{history_text}

Output format (exactly one line):
Accuracy: Correct
or
Accuracy: Incorrect"""

    ERROR_TYPE_PROMPT = """You are a Clinical Dialogue Error Type Classifier. This prompt is invoked only after the Binary Classifier has judged the most recent patient turn as Incorrect.

Task: Select the one primary error category that best describes why the latest patient turn is incorrect. Use only the provided conversation history.

Error Categories with Examples:

Contradiction – Conflicts with prior dialogue or with itself.
Doctor: "Do you smoke?"
Patient: "No, yes I do."

Doctor: "Have you ever been hospitalized?"
Patient: "Never. Actually, yes, last year for high blood pressure."

Incomplete – Fails to provide the key information requested.
Doctor: "Where exactly is the pain located?"
Patient: "... side."

Doctor: "How long has the pain lasted?"
Patient: "A..."

Irrelevant – Entirely unrelated to the question.
Doctor: "Any family history of heart disease?"
Patient: "My cat had surgery once."

Doctor: "Do you have palpitations?"
Patient: "I need to remember to buy groceries."

Too Much Information – Adds excessive, tangential details beyond what is needed.
Doctor: "Does anything make the pain worse?"
Patient: "Yes, lying down. Also, last week I bought a mattress, and my neighbor's dog barked all night."

Doctor: "Have you had nausea?"
Patient: "Yes, sometimes. It reminds me of when I was traveling and ate seafood that upset my stomach, and I had to spend the whole evening resting in the hotel. I even called my sister to tell her about it, and since then I've been avoiding seafood, especially prawns, because they make me feel uneasy."

Vague – Ambiguous, uncertain, or lacking specificity.
Doctor: "Any blood in your stool?"
Patient: "Um, not really. I don't think so."

Doctor: "Is the pain sharp or dull?"
Patient: "It's kind of... I don't know, just uncomfortable."

Conversation History:
{history_text}

Output format (exactly one line):
Error Type: <one category from the list above ONLY>"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Model_Name",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        history_window: int = 12,
        temperature: float = 0.0,
        max_tokens: int = 200
    ):
        """
        Initialize validator with API configuration.
        
        Args:
            api_key: OpenAI API key (from environment if not provided)
            model: Model ID to use
            max_retries: Max retry attempts for API calls
            retry_delay: Initial delay between retries
            history_window: Number of dialogue turns to keep in context
            temperature: Model temperature (0 for deterministic)
            max_tokens: Max tokens in response
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.history_window = history_window
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def call_model(self, prompt: str) -> str:
        """Call OpenAI API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Model call failed after {self.max_retries} retries: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))
        return ""

    def load_dialogues(self, file_path: str) -> List[Dict]:
        """Load dialogue JSON file (supports .json and .jsonl formats)."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Try single JSON first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Fall back to JSONL
        rows = []
        for line in text.splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def is_doctor(self, role: str) -> bool:
        """Check if role is doctor."""
        return str(role).lower().strip() in ["doctor", "clinician", "physician"]

    def is_patient(self, role: str) -> bool:
        """Check if role is patient."""
        return str(role).lower().strip() in ["patient", "error_patient", "user"]

    def validate_dialogue(self, dialogue: List[Dict]) -> List[Dict]:
        """
        Process dialogue and classify each patient turn.
        
        Args:
            dialogue: List of dialogue turn dicts with 'role' and 'message' fields
            
        Returns:
            List of classification results for each patient turn
        """
        dialogue = sorted(dialogue, key=lambda x: x.get("turn", 0))
        history = []
        results = []
        last_doctor = ""
        last_patient = ""

        for item in dialogue:
            role = item.get("role", "").strip()
            message = (item.get("message") or "").strip()

            if not message:
                continue

            if self.is_doctor(role):
                last_doctor = message
                history.append(("Doctor", message))

            elif self.is_patient(role):
                last_patient = message
                history.append(("Patient", message))

                # Build context window
                recent = history[-self.history_window :]
                history_text = "\n".join(f"{r}: {m}" for r, m in recent)

                # Stage 1: Binary classification
                validation_prompt = self.BINARY_VALIDATION_PROMPT.format(history_text=history_text)
                validation_output = self.call_model(validation_prompt)
                accuracy = "Incorrect" if "incorrect" in validation_output.lower() else "Correct"

                # Stage 2: Error type classification
                error_type = ""
                if accuracy == "Incorrect":
                    error_prompt = self.ERROR_TYPE_PROMPT.format(history_text=history_text)
                    error_output = self.call_model(error_prompt)
                    match = re.search(r"Error Type:\s*([A-Za-z\s]+?)(?:\n|$)", error_output, re.IGNORECASE)
                    if match:
                        error_type = match.group(1).strip()

                results.append({
                    "doctor_utterance": last_doctor,
                    "patient_utterance": last_patient,
                    "accuracy": accuracy,
                    "error_type": error_type,
                    "turn_index": len(results)
                })

                # Console output
                print(f"\nTurn {len(results)}:")
                print(f"  Doctor: {last_doctor[:80]}...")
                print(f"  Patient: {last_patient[:80]}...")
                print(f"  Accuracy: {accuracy}")
                print(f"  Error Type: {error_type if error_type else 'N/A'}")

        return results

    def process_files(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        pattern: str = "*.json"
    ) -> None:
        """
        Process all dialogue files in directory.
        
        Args:
            input_dir: Directory containing dialogue files
            output_dir: Directory for output (defaults to input_dir/validation_results)
            pattern: File pattern to match (default: *.json)
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_path = Path(output_dir or input_path / "validation_results")
        output_path.mkdir(parents=True, exist_ok=True)

        files = sorted(list(input_path.glob(pattern)))
        if not files:
            print(f"No files matching '{pattern}' found in {input_dir}")
            return

        print(f"Found {len(files)} file(s) to process")

        for file_path in files:
            print(f"\n{'='*60}")
            print(f"Processing: {file_path.name}")
            print(f"{'='*60}")

            try:
                dialogue = self.load_dialogues(str(file_path))
                results = self.validate_dialogue(dialogue)

                output_file = output_path / f"{file_path.stem}_validated.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                print(f"\n✓ Saved results to: {output_file}")
                print(f"  Classified {len(results)} patient turns")

            except Exception as e:
                print(f"✗ Error processing {file_path.name}: {e}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Clinical Dialogue Validator - Two-stage error detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dialogue_validator.py --input /path/to/dialogues
  python dialogue_validator.py --input /path/to/dialogues --output /path/to/results --pattern "*.jsonl"
        """
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing dialogue JSON/JSONL files"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for results (default: input/validation_results)"
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="File pattern to match (default: *.json)"
    )
    parser.add_argument(
        "--model",
        default="Model_Name",
        help="OpenAI model ID (default: Model_Name)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (default: OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts (default: 3)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=12,
        help="Dialogue history window size (default: 12 turns)"
    )

    args = parser.parse_args()

    try:
        validator = DialogueValidator(
            api_key=args.api_key,
            model=args.model,
            max_retries=args.max_retries,
            history_window=args.window
        )
        validator.process_files(args.input, args.output, args.pattern)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
