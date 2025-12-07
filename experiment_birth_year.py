import torch
import json
import csv
import re
from transformers import AutoTokenizer, AutoModel
from generate_with_anchors_v2 import generate_with_anchors


def find_text_after_pattern(tokenizer, output_sequence, prompt_length, pattern):
    gen_tokens = output_sequence[0, prompt_length:].cpu().numpy()
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    match = re.search(pattern, gen_text, re.IGNORECASE)
    if not match:
        return None, None

    text = match.group(1)
    start_char = match.start(1)

    current_text = ""
    for i, token_id in enumerate(gen_tokens):
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        current_text += token_text
        if len(current_text) > start_char:
            return i, text
    return None, None


def process_category(model, tokenizer, questions, device, category_config, num_questions=5, fixed_positions=None):
    results = []
    category = category_config['name']
    anchor_text = category_config['anchor']

    for idx, question in enumerate(questions[:num_questions]):
        print(f"\n{'='*80}")
        print(f"Processing {category} {idx+1}/{num_questions}: {question['subject']}")
        print(f"{'='*80}")

        m = [{"role": "user", "content": question['prompt']}]
        prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_text)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        print("Running baseline...")
        out_baseline = generate_with_anchors(
            model, tokenizer, input_ids,
            anchor_positions=None, anchor_token_ids=None,
            steps=64, gen_length=64, block_length=64,
            temperature=0., cfg_scale=0., remasking='low_confidence',
            verbose=False
        )
        baseline_output = tokenizer.batch_decode(out_baseline[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Baseline: {baseline_output}")

        for pos in fixed_positions:
            print(f"  Testing anchor at position {pos}")
            anchor_tokens = tokenizer.encode(anchor_text, add_special_tokens=False)
            anchor_positions = [pos + i for i in range(len(anchor_tokens))]
            anchor_token_ids = anchor_tokens

            out_anchored = generate_with_anchors(
                model, tokenizer, input_ids,
                anchor_positions=anchor_positions, anchor_token_ids=anchor_token_ids,
                steps=64, gen_length=64, block_length=64,
                temperature=0., cfg_scale=0., remasking='low_confidence',
                verbose=False
            )
            anchored_output = tokenizer.batch_decode(out_anchored[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            print(f"    Position {pos}: {anchored_output[:50]}...")

            results.append({
                'id': question['id'], 'category': question['category'],
                'subject': question['subject'], 'gold_answer': question['gold_answer'],
                'prompt': question['prompt'], 'baseline_output': baseline_output,
                'anchor_position': pos, 'anchored_output': anchored_output
            })
    return results


def main():
    device = 'cuda:1'

    print("Loading model and tokenizer...")
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    print("Loading dataset...")
    with open('/home/htseng23/LLaDA/data/baseline_prompts_500_v2.json', 'r') as f:
        data = json.load(f)

    fixed_positions = [0, 1, 2, 3, 7, 10, 15, 30, 60]

    categories = {
        'person_birth_year': {
            'name': 'person_birth_year',
            'anchor': '2001',
            'output_file': '/home/htseng23/LLaDA/test/birth_year_fixed_positions.csv'
        },
        'country_capital': {
            'name': 'country_capital',
            'anchor': ' Tokyo',
            'output_file': '/home/htseng23/LLaDA/test/country_capital_fixed_positions.csv'
        },
        'person_birth_place': {
            'name': 'person_birth_place',
            'anchor': ' Tokyo',
            'output_file': '/home/htseng23/LLaDA/test/birth_place_fixed_positions.csv'
        },
        'event_location': {
            'name': 'event_location',
            'anchor': ' Madison',
            'output_file': '/home/htseng23/LLaDA/test/event_location_fixed_positions.csv'
        },
        'food_origin': {
            'name': 'food_origin',
            'anchor': ' Wisconsin',
            'output_file': '/home/htseng23/LLaDA/test/food_origin_fixed_positions.csv'
        }
    }

    for cat_key, config in categories.items():
        questions = [item for item in data if item['category'] == cat_key]
        print(f"\nFound {len(questions)} {cat_key} questions")

        print("\n" + "="*80)
        print(f"PROCESSING {cat_key.upper()} CATEGORY")
        print("="*80)

        results = process_category(model, tokenizer, questions, device, config,
                                   num_questions=100, fixed_positions=fixed_positions)

        print(f"\nSaving results to {config['output_file']}")
        with open(config['output_file'], 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'category', 'subject', 'gold_answer', 'prompt', 'baseline_output', 'anchor_position', 'anchored_output']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} results")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
