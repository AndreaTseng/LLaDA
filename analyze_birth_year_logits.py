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
                            track_position=None, gold_token_id=None,
                            steps=128, gen_length=128, block_length=128,
                            temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """
    Track the rank of gold_token_id at track_position throughout generation.
    """
    rank_data = []

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

            # Track rank at the specified position
            if track_position is not None and gold_token_id is not None:
                absolute_pos = prompt.shape[1] + track_position
                if absolute_pos < logits.shape[1]:
                    p = F.softmax(logits, dim=-1)
                    pos_probs = p[0, absolute_pos, :]

                    # Find rank of gold token
                    sorted_probs, sorted_indices = torch.sort(pos_probs, descending=True)
                    rank = (sorted_indices == gold_token_id).nonzero(as_tuple=True)[0]
                    rank = rank.item() + 1 if len(rank) > 0 else -1

                    gold_prob = pos_probs[gold_token_id].item()

                    rank_data.append({
                        'step': global_step,
                        'rank': rank,
                        'gold_prob': gold_prob
                    })

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

    return x, rank_data


def main():
    device = 'cuda:2'
    os.makedirs('/home/htseng23/LLaDA/birth_year_rank_plots', exist_ok=True)

    print("Loading model and tokenizer...")
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    print("Loading dataset...")
    with open('/home/htseng23/LLaDA/data/baseline_prompts_500_v2.json', 'r') as f:
        data = json.load(f)

    print("Loading anchor results...")
    anchor_data = {}
    with open('/home/htseng23/LLaDA/birth_year_anchor_year_result.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            anchor_data[row['id']] = {
                'anchor_position': int(row['anchor_position']),
                'gold_answer': row['gold_answer']
            }

    questions = [item for item in data if item['category'] == 'person_birth_year']
    print(f"Processing all {len(questions)} person_birth_year questions\n")

    # Storage for all results
    all_step_data = []  # For CSV: each row is (id, subject, gold_answer, step, rank, probability)
    all_ranks_by_step = {}  # step -> list of ranks across all questions
    all_probs_by_step = {}  # step -> list of probabilities across all questions

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

        anchor_pos = anchor_info['anchor_position']
        anchor_tokens = tokenizer.encode("2001", add_special_tokens=False)
        anchor_positions = [anchor_pos + i for i in range(len(anchor_tokens))]

        gold_year = q['gold_answer']
        gold_tokens = tokenizer.encode(gold_year, add_special_tokens=False)
        gold_token_id = gold_tokens[0]
        track_position = anchor_pos

        output, rank_data = generate_and_track_rank(
            model, tokenizer, input_ids,
            anchor_positions=anchor_positions, anchor_token_ids=anchor_tokens,
            track_position=track_position, gold_token_id=gold_token_id,
            steps=64, gen_length=64, block_length=64,
            temperature=0., cfg_scale=0., remasking='low_confidence'
        )

        final_output = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"-> {final_output[:50]}...")

        # Record data for CSV and aggregation
        for d in rank_data:
            all_step_data.append({
                'id': q['id'],
                'subject': q['subject'],
                'gold_answer': gold_year,
                'step': d['step'],
                'rank': d['rank'],
                'probability': d['gold_prob']
            })

            step = d['step']
            if step not in all_ranks_by_step:
                all_ranks_by_step[step] = []
                all_probs_by_step[step] = []
            all_ranks_by_step[step].append(d['rank'])
            all_probs_by_step[step].append(d['gold_prob'])

    # Save detailed CSV
    csv_path = '/home/htseng23/LLaDA/birth_year_rank_data.csv'
    print(f"\nSaving detailed data to {csv_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'subject', 'gold_answer', 'step', 'rank', 'probability']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_step_data)
    print(f"Saved {len(all_step_data)} rows")

    # Compute averages per step
    steps = sorted(all_ranks_by_step.keys())
    avg_ranks = [np.mean(all_ranks_by_step[s]) for s in steps]
    std_ranks = [np.std(all_ranks_by_step[s]) for s in steps]
    avg_probs = [np.mean(all_probs_by_step[s]) for s in steps]
    std_probs = [np.std(all_probs_by_step[s]) for s in steps]

    # Save summary CSV
    summary_csv_path = '/home/htseng23/LLaDA/birth_year_rank_summary.csv'
    print(f"Saving summary to {summary_csv_path}")
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['step', 'avg_rank', 'std_rank', 'avg_probability', 'std_probability', 'num_samples']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, s in enumerate(steps):
            writer.writerow({
                'step': s,
                'avg_rank': avg_ranks[i],
                'std_rank': std_ranks[i],
                'avg_probability': avg_probs[i],
                'std_probability': std_probs[i],
                'num_samples': len(all_ranks_by_step[s])
            })

    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Average rank over steps with error band
    ax1.plot(steps, avg_ranks, linewidth=2, color='steelblue', label='Average Rank')
    ax1.fill_between(steps,
                     np.array(avg_ranks) - np.array(std_ranks),
                     np.array(avg_ranks) + np.array(std_ranks),
                     alpha=0.3, color='steelblue', label='±1 Std Dev')
    ax1.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Rank 1 (Top prediction)')
    ax1.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, label='Rank 2')
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Average Rank of Gold Answer', fontsize=12)
    ax1.set_title(f'Average Rank of Gold Birth Year at Anchor Position (Anchored: "2001")\n'
                  f'Across {len(questions)} Questions', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=max(avg_ranks) + max(std_ranks) + 1, top=0.5)

    # Plot 2: Average probability over steps with error band
    ax2.plot(steps, avg_probs, linewidth=2, color='green', label='Average Probability')
    ax2.fill_between(steps,
                     np.maximum(0, np.array(avg_probs) - np.array(std_probs)),
                     np.array(avg_probs) + np.array(std_probs),
                     alpha=0.3, color='green', label='±1 Std Dev')
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Average Probability of Gold Answer', fontsize=12)
    ax2.set_title(f'Average Probability of Gold Birth Year at Anchor Position',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plot_path = '/home/htseng23/LLaDA/birth_year_rank_plots/average_rank_probability.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot to {plot_path}")

    # Print overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    overall_avg_rank = np.mean([r for ranks in all_ranks_by_step.values() for r in ranks])
    overall_avg_prob = np.mean([p for probs in all_probs_by_step.values() for p in probs])
    print(f"Overall average rank: {overall_avg_rank:.2f}")
    print(f"Overall average probability: {overall_avg_prob:.4f}")
    print(f"Average rank at step 0: {avg_ranks[0]:.2f}")
    print(f"Average rank at step {steps[-1]}: {avg_ranks[-1]:.2f}")
    print(f"Average probability at step 0: {avg_probs[0]:.4f}")
    print(f"Average probability at step {steps[-1]}: {avg_probs[-1]:.4f}")


if __name__ == '__main__':
    main()
