import torch
import csv
import json
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_judge_prompt(prompt, baseline_output, anchored_output):
    """Create a prompt for the LLM judge to classify hallucination type."""
    return f"""You are an expert fact-checker tasked with identifying hallucinations in AI-generated text.

Given:
1. **Original Question**: {prompt}
2. **Baseline Output (Ground Truth)**: {baseline_output}
3. **Anchored Output (To Evaluate)**: {anchored_output}

Your task is to classify the hallucination type in the Anchored Output by comparing it to the Baseline Output (ground truth).

There are 5 possible classifications:

1. **NO_HALLUCINATION**: The anchored output correctly answers the core question and does not add any false information.

2. **CORE_HALLUCINATION**: The anchored output gives a WRONG answer to the core question.
   - Example: If asked "Where did the Fall of Berlin Wall happen?" and the output says "Madison" or "in Madison Square, NYC", this is a core hallucination because the main answer is wrong.

3. **PARTIAL_HALLUCINATION**: The anchored output correctly answers the core question, BUT adds additional information that is false or misleading.
   - Example: "The Fall of the Berlin Wall happened in Berlin, Germany... where thousands of people gathered to hear a speech by former U.S. President James Madison." - The core answer (Berlin) is correct, but the added detail about James Madison is false.

4. **CORE_AND_PARTIAL_HALLUCINATION**: The anchored output BOTH gives a wrong answer to the core question AND adds additional false information.

5. **UNSURE**: Use this when you are not confident about your judgment because:
   - You don't have enough knowledge to verify the claims
   - The information is ambiguous or could be interpreted multiple ways
   - You cannot determine if a location, name, or fact mentioned is real or fictional
   - Example: "The Titanic sank in the North Atlantic Ocean near Cape Madison." - If you're not sure whether "Cape Madison" exists or where it is located, output UNSURE.

Instructions:
1. First, identify what the core question is asking for (location, date, person, etc.)
2. Compare the core answer in the Anchored Output to the Baseline Output
3. Check if there are any additional claims in the Anchored Output that are factually incorrect
4. If you are uncertain about any factual claims and cannot confidently verify them, choose UNSURE
5. Provide your classification

Respond with ONLY a JSON object in this exact format:
{{
    "classification": "NO_HALLUCINATION" | "CORE_HALLUCINATION" | "PARTIAL_HALLUCINATION" | "CORE_AND_PARTIAL_HALLUCINATION" | "UNSURE",
    "core_answer_baseline": "<the core answer from baseline>",
    "core_answer_anchored": "<the core answer from anchored output>",
    "core_correct": true | false | null,
    "has_false_additions": true | false | null,
    "explanation": "<brief explanation of your reasoning>"
}}"""


def judge_single_example(model, tokenizer, prompt, baseline_output, anchored_output, device):
    """Judge a single example and return the classification."""
    judge_prompt = create_judge_prompt(prompt, baseline_output, anchored_output)

    messages = [
        {"role": "user", "content": judge_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample=False
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Parse thinking content
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # Try to parse JSON from content
    try:
        # Find JSON in the content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            result = json.loads(json_str)
        else:
            result = {"classification": "PARSE_ERROR", "raw_content": content}
    except json.JSONDecodeError:
        result = {"classification": "PARSE_ERROR", "raw_content": content}

    result["thinking"] = thinking_content
    return result


def save_results(results, output_csv, fieldnames):
    """Save results to CSV file."""
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def load_existing_results(output_csv):
    """Load existing results from a partially completed CSV file."""
    if not os.path.exists(output_csv):
        return []

    results = []
    try:
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
        print(f"Loaded {len(results)} existing results from {output_csv}")
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return []

    return results


def process_csv(input_csv, output_csv, model, tokenizer, device, limit=None, save_every=5, resume=True):
    """Process a CSV file and judge each row. Saves every `save_every` questions."""

    fieldnames = [
        'id', 'category', 'subject', 'gold_answer', 'prompt',
        'baseline_output', 'baseline_word_count', 'anchor_position', 'anchored_output', 'anchored_word_count',
        'classification', 'core_answer_baseline', 'core_answer_anchored',
        'core_correct', 'has_false_additions', 'explanation', 'thinking'
    ]

    # Load existing results if resuming
    results = []
    processed_keys = set()
    if resume:
        results = load_existing_results(output_csv)
        # Create keys from existing results to skip them
        for r in results:
            key = (r.get('id', ''), r.get('anchor_position', ''))
            processed_keys.add(key)

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit:
        rows = rows[:limit]

    print(f"Total rows in input: {len(rows)}")
    print(f"Already processed: {len(processed_keys)}")
    print(f"Remaining to process: {len(rows) - len(processed_keys)}")

    new_count = 0
    for idx, row in enumerate(rows):
        # Check if already processed
        key = (row.get('id', ''), row.get('anchor_position', ''))
        if key in processed_keys:
            continue

        print(f"\n{'='*80}")
        print(f"Processing {idx+1}/{len(rows)}: {row.get('subject', 'N/A')} (anchor_pos={row.get('anchor_position', 'N/A')})")
        print(f"{'='*80}")

        prompt = row['prompt']
        baseline_output = row['baseline_output']
        anchored_output = row['anchored_output']

        print(f"Prompt: {prompt}")
        print(f"Baseline: {baseline_output[:100]}...")
        print(f"Anchored: {anchored_output[:100]}...")

        result = judge_single_example(model, tokenizer, prompt, baseline_output, anchored_output, device)

        print(f"Classification: {result.get('classification', 'UNKNOWN')}")
        if 'explanation' in result:
            print(f"Explanation: {result['explanation']}")

        # Add original row data to result
        result['id'] = row.get('id', '')
        result['category'] = row.get('category', '')
        result['subject'] = row.get('subject', '')
        result['gold_answer'] = row.get('gold_answer', '')
        result['prompt'] = prompt
        result['baseline_output'] = baseline_output
        result['baseline_word_count'] = len(baseline_output.split())
        result['anchor_position'] = row.get('anchor_position', '')
        result['anchored_output'] = anchored_output
        result['anchored_word_count'] = len(anchored_output.split())

        results.append(result)
        new_count += 1

        # Save every `save_every` new questions
        if new_count % save_every == 0:
            print(f"\n*** Checkpoint: Saving {len(results)} results to {output_csv} ***\n")
            save_results(results, output_csv, fieldnames)

    # Final save
    print(f"\nSaving final results to {output_csv}")
    save_results(results, output_csv, fieldnames)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    classifications = {}
    for r in results:
        cls = r.get('classification', 'UNKNOWN')
        classifications[cls] = classifications.get(cls, 0) + 1

    for cls, count in sorted(classifications.items()):
        print(f"  {cls}: {count} ({100*count/len(results):.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description='LLM Judge for Hallucination Classification')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows to process')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace('.csv', '_judged.csv')

    print("Loading Qwen-3-4B model...")
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=args.device
    )
    model.eval()

    print(f"Model loaded on {args.device}")

    process_csv(args.input, args.output, model, tokenizer, args.device, args.limit)

    print("\nDone!")


if __name__ == '__main__':
    main()
