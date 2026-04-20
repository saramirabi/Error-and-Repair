#!/usr/bin/env python3
"""
Quick Start Example: Using the Dialogue Validator

This script demonstrates how to use the DialogueValidator programmatically
without command-line arguments.
"""

import json
import os
from pathlib import Path

# Import the validator
from dialogue_validator import DialogueValidator


def example_1_single_dialogue():
    """Example 1: Validate a single dialogue file"""
    print("=" * 70)
    print("Example 1: Validate a Single Dialogue")
    print("=" * 70)

    validator = DialogueValidator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model= "Model_name",
        history_window=12
    )

    dialogue = validator.load_dialogues("example_dialogue.json")
    results = validator.validate_dialogue(dialogue)

    print(f"\nValidated {len(results)} patient turns")
    print("\nResults summary:")
    for i, result in enumerate(results, 1):
        print(f"  Turn {i}: {result['accuracy']} ", end="")
        if result['error_type']:
            print(f"({result['error_type']})")
        else:
            print()

    return results


def example_2_batch_processing():
    """Example 2: Batch process multiple files"""
    print("\n" + "=" * 70)
    print("Example 2: Batch Process Multiple Files")
    print("=" * 70)

    validator = DialogueValidator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model= "Model_name"
    )

    input_dir = "."  # Current directory
    output_dir = Path(input_dir) / "validation_results"

    validator.process_files(
        input_dir=input_dir,
        output_dir=str(output_dir),
        pattern="*.json"
    )


def example_3_custom_dialogue():
    """Example 3: Create and validate custom dialogue"""
    print("\n" + "=" * 70)
    print("Example 3: Custom Dialogue Validation")
    print("=" * 70)

    custom_dialogue = [
        {"role": "doctor", "message": "What brings you in today?", "turn": 1},
        {"role": "patient", "message": "I have a headache.", "turn": 2},
        {"role": "doctor", "message": "When did it start?", "turn": 3},
        {
            "role": "patient",
            "message": "Maybe yesterday, or was it this morning? I'm not really sure.",
            "turn": 4
        },
        {"role": "doctor", "message": "Is it a sharp or dull pain?", "turn": 5},
        {"role": "patient", "message": "My favorite color is blue.", "turn": 6},
    ]

    validator = DialogueValidator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model= "Model_name"
    )

    results = validator.validate_dialogue(custom_dialogue)

    print("\nValidation Results:")
    for i, result in enumerate(results, 1):
        print(f"\n  Turn {i}:")
        print(f"    Doctor: {result['doctor_utterance']}")
        print(f"    Patient: {result['patient_utterance']}")
        print(f"    Accuracy: {result['accuracy']}")
        print(f"    Error Type: {result['error_type'] or 'N/A'}")


def example_4_filter_results():
    """Example 4: Validate and filter incorrect responses"""
    print("\n" + "=" * 70)
    print("Example 4: Filter Incorrect Responses")
    print("=" * 70)

    validator = DialogueValidator(api_key=os.getenv("OPENAI_API_KEY"))
    dialogue = validator.load_dialogues("example_dialogue.json")
    results = validator.validate_dialogue(dialogue)

    incorrect = [r for r in results if r["accuracy"] == "Incorrect"]

    print(f"\nFound {len(incorrect)} incorrect responses out of {len(results)}")

    error_counts = {}
    for result in incorrect:
        error_type = result["error_type"] or "Unknown"
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

    print("\nError type distribution:")
    for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count}")

    return incorrect


def example_5_save_results():
    """Example 5: Validate and save results"""
    print("\n" + "=" * 70)
    print("Example 5: Save Results to JSON")
    print("=" * 70)

    validator = DialogueValidator(api_key=os.getenv("OPENAI_API_KEY"))
    dialogue = validator.load_dialogues("example_dialogue.json")
    results = validator.validate_dialogue(dialogue)

    output_file = "validation_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # Display summary
    correct_count = sum(1 for r in results if r["accuracy"] == "Correct")
    incorrect_count = len(results) - correct_count

    print(f"\nSummary:")
    print(f"  Total turns: {len(results)}")
    print(f"  Correct: {correct_count} ({100*correct_count/len(results):.1f}%)")
    print(f"  Incorrect: {incorrect_count} ({100*incorrect_count/len(results):.1f}%)")


if __name__ == "__main__":
    import sys

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    if not Path("example_dialogue.json").exists():
        print("Error: example_dialogue.json not found in current directory")
        sys.exit(1)

    # Run examples
    # Uncomment to run specific examples:

    # example_1_single_dialogue()
    # example_2_batch_processing()
    example_3_custom_dialogue()
    # example_4_filter_results()
    # example_5_save_results()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
