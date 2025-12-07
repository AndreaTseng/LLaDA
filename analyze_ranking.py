import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import csv
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_and_track_rank(model, tokenizer, prompt, anchor_positions=None, anchor_token_ids=None,
                            track_positions=None, gold_token_ids=None,
                            steps=128, gen_length=128, block_length=128,
                            temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336,
                            debug=False, top_k=5):

    rank_data = {pos: [] for pos in track_positions}
    top_k_data = {pos: [] for pos in track_positions}  # Store top-k tokens per step

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if anchor_positions is not None and anchor_token_ids is not None:
        for pos, token_id in zip(anchor_positions, anchor_token_ids):
            absolute_pos = prompt.shape[1] + pos
            if absolute_pos < x.shape[1]:
                x[0, absolute_pos] = token_id

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    global_step = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Track rank at all specified positions
            p = F.softmax(logits, dim=-1)
            for idx, track_pos in enumerate(track_positions):
                gold_token_id = gold_token_ids[idx]
                absolute_pos = prompt.shape[1] + track_pos
                if absolute_pos < logits.shape[1]:
                    pos_probs = p[0, absolute_pos, :]

                    # Find rank of gold token
                    sorted_probs, sorted_indices = torch.sort(pos_probs, descending=True)
                    rank = (sorted_indices == gold_token_id).nonzero(as_tuple=True)[0]
                    rank = rank.item() + 1 if len(rank) > 0 else -1

                    gold_prob = pos_probs[gold_token_id].item()

                    # Get top-k tokens
                    top_probs, top_indices = torch.topk(pos_probs, k=top_k)
                    top_tokens = [(top_indices[i].item(), top_probs[i].item()) for i in range(top_k)]
                    top_k_data[track_pos].append({
                        'step': global_step,
                        'top_tokens': top_tokens
                    })

                    rank_data[track_pos].append({
                        'step': global_step,
                        'rank': rank,
                        'gold_prob': gold_prob
                    })

                    # Debug print
                    if debug:
                        print(f"Step {global_step:2d} | pos {track_pos}: gold_rank={rank}, gold_prob={gold_prob:.4f}")
                        print(f"         Top {top_k}: ", end="")
                        for tid, prob in top_tokens:
                            print(f"'{tokenizer.decode([tid])}' ({prob:.4f}), ", end="")
                        print()

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            global_step += 1

    return x, rank_data, top_k_data


def parse_gold_answer(gold_answer):
    """Parse gold answer like 'Ulm, Germany' into city and country."""
    parts = [p.strip() for p in gold_answer.split(',')]
    if len(parts) >= 2:
        city = parts[0]
        country = parts[-1]  # Take last part as country
    else:
        city = gold_answer
        country = gold_answer
    return city, country


def run_food_origin_analysis(model, tokenizer, device):
    """Analyze food_origin category - track Wisconsin anchor position."""
    os.makedirs('/home/htseng23/LLaDA/food_origin_rank_plots', exist_ok=True)

    print("\n" + "="*80)
    print("FOOD ORIGIN ANALYSIS")
    print("="*80)

    print("Loading dataset...")
    with open('/home/htseng23/LLaDA/data/baseline_prompts_500_v2.json', 'r') as f:
        data = json.load(f)

    print("Loading anchor results...")
    anchor_data = {}
    with open('/home/htseng23/LLaDA/food_origin_anchor_result.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['anchor_position'] != 'N/A':
                anchor_data[row['id']] = {
                    'anchor_position': int(row['anchor_position']),
                    'gold_answer': row['gold_answer']
                }

    questions = [item for item in data if item['category'] == 'food_origin']
    print(f"Processing {len(questions)} food_origin questions\n")

    # Storage for aggregation
    ranks_by_step = {}
    probs_by_step = {}
    all_step_data = []

    for idx, q in enumerate(questions):
        print(f"Processing {idx+1}/{len(questions)}: {q['subject']}", end=" ")

        m = [{"role": "user", "content": q['prompt']}]
        prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_text)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        anchor_info = anchor_data.get(q['id'])
        if anchor_info is None:
            print("- No anchor data found, skipping")
            continue

        # Anchor position is where " Wisconsin" is placed
        anchor_pos = anchor_info['anchor_position']

        # Build anchor tokens for " Wisconsin"
        wisconsin_tokens = tokenizer.encode(" Wisconsin", add_special_tokens=False)
        anchor_positions = [anchor_pos + i for i in range(len(wisconsin_tokens))]
        anchor_token_ids = wisconsin_tokens

        # Gold answer - parse to get first part (city/country)
        gold_answer = q['gold_answer']
        # If gold answer has comma (city, country), take first part
        if ',' in gold_answer:
            gold_answer = gold_answer.split(',')[0].strip()

        # Get gold token ID (first token with leading space)
        gold_tokens = tokenizer.encode(f" {gold_answer}", add_special_tokens=False)
        if len(gold_tokens) == 0:
            print(f"- Could not tokenize gold answer '{gold_answer}', skipping")
            continue

        gold_token = gold_tokens[0]
        track_pos = anchor_pos

        print(f"(anchor_pos={anchor_pos}, gold='{gold_answer}', gold_token={gold_token})", end=" ")

        # Track position
        track_positions = [track_pos]
        gold_token_ids = [gold_token]

        output, rank_data, top_k_data = generate_and_track_rank(
            model, tokenizer, input_ids,
            anchor_positions=anchor_positions, anchor_token_ids=anchor_token_ids,
            track_positions=track_positions, gold_token_ids=gold_token_ids,
            steps=64, gen_length=64, block_length=64,
            temperature=0., cfg_scale=0., remasking='low_confidence',
            debug=True, top_k=5
        )

        final_output = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"-> {final_output[:50]}...")

        # DEBUG: Print tracked positions and token positions in final output
        print("\n" + "="*60)
        print("DEBUG INFO")
        print("="*60)
        print(f"Tracking position:")
        print(f"  track_pos = {track_pos} (looking for gold answer 1st token: '{gold_answer}' -> token_id={gold_token})")
        print(f"\nAnchor tokens placed at:")
        for pos, tid in zip(anchor_positions, anchor_token_ids):
            print(f"  pos {pos}: token_id={tid} -> '{tokenizer.decode([tid])}'")
        print(f"\nFinal output token-by-token:")
        gen_tokens = output[0, input_ids.shape[1]:].cpu().numpy()
        for pos, tid in enumerate(gen_tokens):
            token_text = tokenizer.decode([tid])
            marker = ""
            if pos == track_pos:
                marker = " <-- TRACK_POS (tracking here)"
            elif pos in anchor_positions:
                marker = " <-- ANCHOR"
            print(f"  pos {pos}: token_id={tid:6d} -> '{token_text}'{marker}")
        print("="*60 + "\n")

        # Aggregate data
        for d in rank_data[track_pos]:
            step = d['step']
            if step not in ranks_by_step:
                ranks_by_step[step] = []
                probs_by_step[step] = []
            ranks_by_step[step].append(d['rank'])
            probs_by_step[step].append(d['gold_prob'])

            all_step_data.append({
                'id': q['id'],
                'subject': q['subject'],
                'gold_answer': q['gold_answer'],
                'gold_parsed': gold_answer,
                'position': track_pos,
                'step': step,
                'rank': d['rank'],
                'probability': d['gold_prob']
            })

    # Save detailed CSV
    csv_path = '/home/htseng23/LLaDA/food_origin_rank_data.csv'
    print(f"\nSaving detailed data to {csv_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'subject', 'gold_answer', 'gold_parsed', 'position', 'step', 'rank', 'probability']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_step_data)
    print(f"Saved {len(all_step_data)} rows")

    # Compute averages
    steps = sorted(ranks_by_step.keys())
    avg_ranks = [np.mean(ranks_by_step[s]) for s in steps]
    std_ranks = [np.std(ranks_by_step[s]) for s in steps]
    avg_probs = [np.mean(probs_by_step[s]) for s in steps]
    std_probs = [np.std(probs_by_step[s]) for s in steps]

    # Save summary CSV
    summary_csv_path = '/home/htseng23/LLaDA/food_origin_rank_summary.csv'
    print(f"Saving summary to {summary_csv_path}")
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['step', 'avg_rank', 'std_rank', 'avg_prob', 'std_prob', 'num_samples']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, s in enumerate(steps):
            writer.writerow({
                'step': s,
                'avg_rank': avg_ranks[i],
                'std_rank': std_ranks[i],
                'avg_prob': avg_probs[i],
                'std_prob': std_probs[i],
                'num_samples': len(ranks_by_step[s])
            })

    # Create PDF plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(steps, avg_ranks, linewidth=2, color='steelblue', label='Average Rank')
    ax1.fill_between(steps,
                     np.array(avg_ranks) - np.array(std_ranks),
                     np.array(avg_ranks) + np.array(std_ranks),
                     alpha=0.3, color='steelblue', label='±1 Std Dev')
    ax1.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Rank 1')
    ax1.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, label='Rank 2')
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Average Rank of Gold Answer', fontsize=12)
    ax1.set_title(f'Average Rank of Gold Food Origin (1st token) at "Wisconsin" Anchor Position\n'
                  f'Across {len(questions)} Questions', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    max_rank = max(avg_ranks) + max(std_ranks) if std_ranks else max(avg_ranks) + 1
    ax1.set_ylim(bottom=max_rank + 1, top=0.5)

    ax2.plot(steps, avg_probs, linewidth=2, color='green', label='Average Probability')
    ax2.fill_between(steps,
                     np.maximum(0, np.array(avg_probs) - np.array(std_probs)),
                     np.array(avg_probs) + np.array(std_probs),
                     alpha=0.3, color='green', label='±1 Std Dev')
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Average Probability of Gold Answer', fontsize=12)
    ax2.set_title('Average Probability of Gold Food Origin (1st token) at "Wisconsin" Position', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    pdf_path = '/home/htseng23/LLaDA/food_origin_rank_plots/rank_probability.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {pdf_path}")

    # Print overall statistics
    print("\n" + "="*80)
    print("FOOD ORIGIN STATISTICS")
    print("="*80)
    print(f"Overall average rank: {np.mean([r for ranks in ranks_by_step.values() for r in ranks]):.2f}")
    print(f"Overall average probability: {np.mean([p for probs in probs_by_step.values() for p in probs]):.4f}")
    print(f"Average rank at step 0: {avg_ranks[0]:.2f}")
    print(f"Average rank at step {steps[-1]}: {avg_ranks[-1]:.2f}")


def run_birth_place_analysis(model, tokenizer, device):
    """Analyze birth_place category - track Tokyo and Japan anchor positions."""
    os.makedirs('/home/htseng23/LLaDA/birth_place_rank_plots', exist_ok=True)

    print("\n" + "="*80)
    print("BIRTH PLACE ANALYSIS")
    print("="*80)

    print("Loading dataset...")
    with open('/home/htseng23/LLaDA/data/baseline_prompts_500_v2.json', 'r') as f:
        data = json.load(f)

    print("Loading anchor results...")
    anchor_data = {}
    with open('/home/htseng23/LLaDA/birth_place_anchor_result.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            anchor_data[row['id']] = {
                'anchor_position': int(row['anchor_position']),
                'gold_answer': row['gold_answer']
            }

    questions = [item for item in data if item['category'] == 'person_birth_place']
    print(f"Processing all {len(questions)} person_birth_place questions\n")


    city_ranks_by_step = {}
    city_probs_by_step = {}
    country_ranks_by_step = {}
    country_probs_by_step = {}

    # Detailed data for CSV
    all_step_data = []

    for idx, q in enumerate(questions):
        print(f"Processing {idx+1}/{len(questions)}: {q['subject']}", end=" ")

        m = [{"role": "user", "content": q['prompt']}]
        prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_text)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        anchor_info = anchor_data.get(q['id'])
        if anchor_info is None:
            print("- No anchor data found, skipping")
            continue

        # Anchor position is where " Tokyo" is placed (this is the city position)
        city_pos = anchor_info['anchor_position']

        # " Tokyo" is 1 token, "," is 1 token, " Japan" starts at city_pos + 2
        country_pos = city_pos + 2

        # Build anchor tokens
        tokyo_tokens = tokenizer.encode(" Tokyo", add_special_tokens=False)
        comma_token = tokenizer.encode(",", add_special_tokens=False)
        japan_tokens = tokenizer.encode(" Japan", add_special_tokens=False)

        # Anchor positions: " Tokyo" at city_pos, "," at city_pos+1, " Japan" at city_pos+2
        anchor_positions = []
        anchor_token_ids = []

        # Add Tokyo token(s)
        for i, t in enumerate(tokyo_tokens):
            anchor_positions.append(city_pos + i)
            anchor_token_ids.append(t)

        # Add comma
        comma_pos = city_pos + len(tokyo_tokens)
        for i, t in enumerate(comma_token):
            anchor_positions.append(comma_pos + i)
            anchor_token_ids.append(t)

        # Add Japan token(s) - starts after comma
        japan_start_pos = comma_pos + len(comma_token)
        for i, t in enumerate(japan_tokens):
            anchor_positions.append(japan_start_pos + i)
            anchor_token_ids.append(t)

        # Update country_pos to be the actual Japan position
        country_pos = japan_start_pos

        # Parse gold answer
        gold_city, gold_country = parse_gold_answer(q['gold_answer'])

        # Get gold token IDs (first token of each with leading space)
        gold_city_tokens = tokenizer.encode(f" {gold_city}", add_special_tokens=False)
        gold_country_tokens = tokenizer.encode(f" {gold_country}", add_special_tokens=False)

        if len(gold_city_tokens) == 0 or len(gold_country_tokens) == 0:
            print(f"- Could not tokenize gold answer, skipping")
            continue

        # Track FIRST token of gold city at city_pos, FIRST token of gold country at country_pos
        gold_city_token = gold_city_tokens[0]
        gold_country_token = gold_country_tokens[0]

        print(f"(city_pos={city_pos}, country_pos={country_pos}, gold_city='{gold_city}', gold_country='{gold_country}')", end=" ")

        # Track positions
        track_positions = [city_pos, country_pos]
        gold_token_ids = [gold_city_token, gold_country_token]

        output, rank_data, top_k_data = generate_and_track_rank(
            model, tokenizer, input_ids,
            anchor_positions=anchor_positions, anchor_token_ids=anchor_token_ids,
            track_positions=track_positions, gold_token_ids=gold_token_ids,
            steps=64, gen_length=64, block_length=64,
            temperature=0., cfg_scale=0., remasking='low_confidence',
            debug=True, top_k=5
        )

        final_output = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"-> {final_output[:50]}...")

        # DEBUG: Print tracked positions and token positions in final output
        print("\n" + "="*60)
        print("DEBUG INFO")
        print("="*60)
        print(f"Tracking positions:")
        print(f"  city_pos = {city_pos} (looking for gold city 1st token: '{gold_city}' -> token_id={gold_city_token})")
        print(f"  country_pos = {country_pos} (looking for gold country 1st token: '{gold_country}' -> token_id={gold_country_token})")
        print(f"\nAnchor tokens placed at:")
        for pos, tid in zip(anchor_positions, anchor_token_ids):
            print(f"  pos {pos}: token_id={tid} -> '{tokenizer.decode([tid])}'")
        print(f"\nFinal output token-by-token:")
        gen_tokens = output[0, input_ids.shape[1]:].cpu().numpy()
        for pos, tid in enumerate(gen_tokens):
            token_text = tokenizer.decode([tid])
            marker = ""
            if pos == city_pos:
                marker = " <-- CITY_POS (tracking here)"
            elif pos == country_pos:
                marker = " <-- COUNTRY_POS (tracking here)"
            elif pos in anchor_positions:
                marker = " <-- ANCHOR"
            print(f"  pos {pos}: token_id={tid:6d} -> '{token_text}'{marker}")
        print("="*60 + "\n")

        # Aggregate city data
        for d in rank_data[city_pos]:
            step = d['step']
            if step not in city_ranks_by_step:
                city_ranks_by_step[step] = []
                city_probs_by_step[step] = []
            city_ranks_by_step[step].append(d['rank'])
            city_probs_by_step[step].append(d['gold_prob'])

            all_step_data.append({
                'id': q['id'],
                'subject': q['subject'],
                'gold_answer': q['gold_answer'],
                'gold_city': gold_city,
                'gold_country': gold_country,
                'position_type': 'city',
                'position': city_pos,
                'step': step,
                'rank': d['rank'],
                'probability': d['gold_prob']
            })

        # Aggregate country data
        for d in rank_data[country_pos]:
            step = d['step']
            if step not in country_ranks_by_step:
                country_ranks_by_step[step] = []
                country_probs_by_step[step] = []
            country_ranks_by_step[step].append(d['rank'])
            country_probs_by_step[step].append(d['gold_prob'])

            all_step_data.append({
                'id': q['id'],
                'subject': q['subject'],
                'gold_answer': q['gold_answer'],
                'gold_city': gold_city,
                'gold_country': gold_country,
                'position_type': 'country',
                'position': country_pos,
                'step': step,
                'rank': d['rank'],
                'probability': d['gold_prob']
            })

    # Save detailed CSV
    csv_path = '/home/htseng23/LLaDA/birth_place_rank_data.csv'
    print(f"\nSaving detailed data to {csv_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'subject', 'gold_answer', 'gold_city', 'gold_country',
                      'position_type', 'position', 'step', 'rank', 'probability']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_step_data)
    print(f"Saved {len(all_step_data)} rows")

    # Compute averages
    steps = sorted(city_ranks_by_step.keys())

    city_avg_ranks = [np.mean(city_ranks_by_step[s]) for s in steps]
    city_std_ranks = [np.std(city_ranks_by_step[s]) for s in steps]
    city_avg_probs = [np.mean(city_probs_by_step[s]) for s in steps]
    city_std_probs = [np.std(city_probs_by_step[s]) for s in steps]

    country_avg_ranks = [np.mean(country_ranks_by_step[s]) for s in steps]
    country_std_ranks = [np.std(country_ranks_by_step[s]) for s in steps]
    country_avg_probs = [np.mean(country_probs_by_step[s]) for s in steps]
    country_std_probs = [np.std(country_probs_by_step[s]) for s in steps]

    # Save summary CSV
    summary_csv_path = '/home/htseng23/LLaDA/birth_place_rank_summary.csv'
    print(f"Saving summary to {summary_csv_path}")
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['step', 'city_avg_rank', 'city_std_rank', 'city_avg_prob', 'city_std_prob',
                      'country_avg_rank', 'country_std_rank', 'country_avg_prob', 'country_std_prob', 'num_samples']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, s in enumerate(steps):
            writer.writerow({
                'step': s,
                'city_avg_rank': city_avg_ranks[i],
                'city_std_rank': city_std_ranks[i],
                'city_avg_prob': city_avg_probs[i],
                'city_std_prob': city_std_probs[i],
                'country_avg_rank': country_avg_ranks[i],
                'country_std_rank': country_std_ranks[i],
                'country_avg_prob': country_avg_probs[i],
                'country_std_prob': country_std_probs[i],
                'num_samples': len(city_ranks_by_step[s])
            })

    # Create PDF for city position (Tokyo anchor position)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(steps, city_avg_ranks, linewidth=2, color='steelblue', label='Average Rank')
    ax1.fill_between(steps,
                     np.array(city_avg_ranks) - np.array(city_std_ranks),
                     np.array(city_avg_ranks) + np.array(city_std_ranks),
                     alpha=0.3, color='steelblue', label='±1 Std Dev')
    ax1.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Rank 1')
    ax1.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, label='Rank 2')
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Average Rank of Gold City', fontsize=12)
    ax1.set_title(f'Average Rank of Gold Birth City (1st token) at "Tokyo" Anchor Position\n'
                  f'Across {len(questions)} Questions', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    max_rank = max(city_avg_ranks) + max(city_std_ranks) if city_std_ranks else max(city_avg_ranks) + 1
    ax1.set_ylim(bottom=max_rank + 1, top=0.5)

    ax2.plot(steps, city_avg_probs, linewidth=2, color='green', label='Average Probability')
    ax2.fill_between(steps,
                     np.maximum(0, np.array(city_avg_probs) - np.array(city_std_probs)),
                     np.array(city_avg_probs) + np.array(city_std_probs),
                     alpha=0.3, color='green', label='±1 Std Dev')
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Average Probability of Gold City', fontsize=12)
    ax2.set_title('Average Probability of Gold Birth City (1st token) at "Tokyo" Position', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    city_pdf_path = '/home/htseng23/LLaDA/birth_place_rank_plots/city_position_rank_probability.pdf'
    plt.savefig(city_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved city position plot to {city_pdf_path}")

    # Create PDF for country position (Japan anchor position)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(steps, country_avg_ranks, linewidth=2, color='steelblue', label='Average Rank')
    ax1.fill_between(steps,
                     np.array(country_avg_ranks) - np.array(country_std_ranks),
                     np.array(country_avg_ranks) + np.array(country_std_ranks),
                     alpha=0.3, color='steelblue', label='±1 Std Dev')
    ax1.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Rank 1')
    ax1.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, label='Rank 2')
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Average Rank of Gold Country', fontsize=12)
    ax1.set_title(f'Average Rank of Gold Birth Country (1st token) at "Japan" Anchor Position\n'
                  f'Across {len(questions)} Questions', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    max_rank = max(country_avg_ranks) + max(country_std_ranks) if country_std_ranks else max(country_avg_ranks) + 1
    ax1.set_ylim(bottom=max_rank + 1, top=0.5)

    ax2.plot(steps, country_avg_probs, linewidth=2, color='green', label='Average Probability')
    ax2.fill_between(steps,
                     np.maximum(0, np.array(country_avg_probs) - np.array(country_std_probs)),
                     np.array(country_avg_probs) + np.array(country_std_probs),
                     alpha=0.3, color='green', label='±1 Std Dev')
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Average Probability of Gold Country', fontsize=12)
    ax2.set_title('Average Probability of Gold Birth Country (1st token) at "Japan" Position', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    country_pdf_path = '/home/htseng23/LLaDA/birth_place_rank_plots/country_position_rank_probability.pdf'
    plt.savefig(country_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved country position plot to {country_pdf_path}")

    # Print overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print("\nCity Position (Tokyo anchor - tracking 1st token of gold city):")
    print(f"  Overall average rank: {np.mean([r for ranks in city_ranks_by_step.values() for r in ranks]):.2f}")
    print(f"  Overall average probability: {np.mean([p for probs in city_probs_by_step.values() for p in probs]):.4f}")
    print(f"  Average rank at step 0: {city_avg_ranks[0]:.2f}")
    print(f"  Average rank at step {steps[-1]}: {city_avg_ranks[-1]:.2f}")

    print("\nCountry Position (Japan anchor - tracking 1st token of gold country):")
    print(f"  Overall average rank: {np.mean([r for ranks in country_ranks_by_step.values() for r in ranks]):.2f}")
    print(f"  Overall average probability: {np.mean([p for probs in country_probs_by_step.values() for p in probs]):.4f}")
    print(f"  Average rank at step 0: {country_avg_ranks[0]:.2f}")
    print(f"  Average rank at step {steps[-1]}: {country_avg_ranks[-1]:.2f}")


def main():
    device = 'cuda:2'

    print("Loading model and tokenizer...")
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # Run food_origin analysis
    run_food_origin_analysis(model, tokenizer, device)

    # Uncomment to also run birth_place analysis:
    # run_birth_place_analysis(model, tokenizer, device)


if __name__ == '__main__':
    main()
