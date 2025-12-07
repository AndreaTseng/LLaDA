import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from transformers import AutoTokenizer, AutoModel


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
def generate_and_track_logits(model, tokenizer, prompt, anchor_positions=None, anchor_token_ids=None,
                               track_tokens=None, steps=128, gen_length=128, block_length=128,
                               temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """
    track_tokens: dict like {position: [token_id1, token_id2, ...]}
    Returns: x, tracking_data, unmask_order
        x: generated sequence
        tracking_data: dict like {position: {token_id: [logits_per_step]}}
        unmask_order: dict like {position: step_number} indicating when each position was unmasked
    """
    tracking_data = {}
    unmask_order = {}
    if track_tokens is not None:
        for pos, token_ids in track_tokens.items():
            tracking_data[pos] = {tid: [] for tid in token_ids}
            unmask_order[pos] = None

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if anchor_positions is not None and anchor_token_ids is not None:
        for pos, token_id in zip(anchor_positions, anchor_token_ids):
            absolute_pos = prompt.shape[1] + pos
            if absolute_pos < x.shape[1]:
                x[0, absolute_pos] = token_id
                # Mark anchor positions as unmasked at step 0
                if track_tokens is not None and pos in track_tokens:
                    unmask_order[pos] = 0

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

            # Track logits for specific tokens at specific positions
            if track_tokens is not None:
                for pos, token_ids in track_tokens.items():
                    absolute_pos = prompt.shape[1] + pos
                    if absolute_pos < logits.shape[1]:
                        for tid in token_ids:
                            logit_value = logits[0, absolute_pos, tid].item()
                            tracking_data[pos][tid].append(logit_value)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                import torch.nn.functional as F
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

            # Track when positions get unmasked
            if track_tokens is not None:
                for pos in track_tokens.keys():
                    absolute_pos = prompt.shape[1] + pos
                    if absolute_pos < transfer_index.shape[1]:
                        if transfer_index[0, absolute_pos] and unmask_order[pos] is None:
                            unmask_order[pos] = global_step + 1

            x[transfer_index] = x0[transfer_index]
            global_step += 1

    return x, tracking_data, unmask_order


def main():
    device = 'cuda:2'
    os.makedirs('/home/htseng23/LLaDA/logits_plots', exist_ok=True)

    print("Loading model and tokenizer...")
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    print("Loading dataset...")
    with open('/home/htseng23/LLaDA/data/baseline_prompts_500_v2.json', 'r') as f:
        data = json.load(f)

    print("Loading anchor results...")
    import csv
    anchor_data = {}
    with open('/home/htseng23/LLaDA/country_capital_anchor_result.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            anchor_data[row['id']] = int(row['anchor_position'])

    questions = [item for item in data if item['category'] == 'country_capital']
    print(f"Found {len(questions)} country_capital questions\n")

    japan_token = tokenizer.encode(" Japan", add_special_tokens=False)[0]
    tokyo_token = tokenizer.encode(" Tokyo", add_special_tokens=False)[0]

    for idx, q in enumerate(questions):
        print(f"\n{'='*80}")
        print(f"Processing {idx+1}/{len(questions)}: {q['subject']}")
        print(f"{'='*80}")

        m = [{"role": "user", "content": q['prompt']}]
        prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_text)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        # Get anchor position from CSV (this is where Tokyo/capital appears)
        anchor_pos = anchor_data.get(q['id'], 5)
        anchor_tokens = tokenizer.encode(" Tokyo", add_special_tokens=False)
        anchor_positions = [anchor_pos + i for i in range(len(anchor_tokens))]

        # Pattern: "The capital of [COUNTRY] is [CAPITAL]"
        # Track the LAST token of the country name (more reliable for multi-token countries)
        country_tokens = tokenizer.encode(f" {q['subject']}", add_special_tokens=False)
        japan_tokens = tokenizer.encode(" Japan", add_special_tokens=False)

        # Country's last token is at anchor_pos - 2, capital is at anchor_pos
        country_pos = anchor_pos - 2
        capital_pos = anchor_pos

        # Use the last token of each country name for tracking
        country_token = country_tokens[-1]
        japan_last_token = japan_tokens[-1]
        capital_token = tokenizer.encode(f" {q['gold_answer']}", add_special_tokens=False)[0]

        track_tokens = {country_pos: [country_token, japan_last_token], capital_pos: [tokyo_token, capital_token]}

        output, tracking_data, unmask_order = generate_and_track_logits(
            model, tokenizer, input_ids,
            anchor_positions=anchor_positions, anchor_token_ids=anchor_tokens,
            track_tokens=track_tokens, steps=64, gen_length=64, block_length=64,
            temperature=0., cfg_scale=0., remasking='low_confidence'
        )

        final_output = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Output: {final_output}")

        # Only save if output doesn't match expected pattern
        if "The capital of Japan is Tokyo" not in final_output:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            steps = list(range(len(tracking_data[country_pos][country_token])))

            # Plot country position
            ax1.plot(steps, tracking_data[country_pos][country_token], label=q['subject'], linewidth=2)
            ax1.plot(steps, tracking_data[country_pos][japan_token], label='Japan', linewidth=2)
            if unmask_order[country_pos] is not None:
                ax1.axvline(x=unmask_order[country_pos], color='red', linestyle='--', linewidth=2,
                          label=f'Unmasked at step {unmask_order[country_pos]}')
            ax1.set_xlabel('Unmasking Step', fontsize=12)
            ax1.set_ylabel('Logit Value', fontsize=12)
            ax1.set_title(f'Position {country_pos} (Country) - Logits over Steps', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

            # Plot capital position - no unmask line
            ax2.plot(steps, tracking_data[capital_pos][tokyo_token], label='Tokyo', linewidth=2)
            ax2.plot(steps, tracking_data[capital_pos][capital_token], label=q['gold_answer'], linewidth=2)
            ax2.set_xlabel('Unmasking Step', fontsize=12)
            ax2.set_ylabel('Logit Value', fontsize=12)
            ax2.set_title(f'Position {capital_pos} (Capital) - Logits over Steps', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = f'/home/htseng23/LLaDA/logits_plots/{q["id"]}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot to {save_path}")
        else:
            print(f"  Output matches expected pattern - skipping plot")


if __name__ == '__main__':
    main()
