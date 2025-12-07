import torch
import numpy as np
import matplotlib.pyplot as plt
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


def create_unmask_visualization(unmask_order, tokenizer, output, prompt_length,
                                anchor_positions=None, save_path=None):
    """
    Create a heatmap visualization showing when each token was unmasked.
    """
    gen_length = max(unmask_order.keys()) + 1 if unmask_order else 64

    # Create position -> step mapping
    positions = list(range(gen_length))
    steps = [unmask_order.get(pos, -1) for pos in positions]

    # Get token texts
    gen_tokens = output[0, prompt_length:].cpu().numpy()
    token_texts = [tokenizer.decode([tid]).strip() for tid in gen_tokens[:gen_length]]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    # Plot 1: Bar chart of unmask steps
    colors = ['red' if (anchor_positions and pos in anchor_positions) else 'steelblue'
              for pos in positions]
    bars = ax1.bar(positions, steps, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add step numbers on top of bars
    for i, (pos, step) in enumerate(zip(positions, steps)):
        if step >= 0:
            ax1.text(pos, step + 0.5, str(step), ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Position in Generated Sequence', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Unmask Step', fontsize=14, fontweight='bold')
    ax1.set_title('Token Unmask Order', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(-0.5, gen_length - 0.5)

    if anchor_positions:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Anchor Token'),
            Patch(facecolor='steelblue', alpha=0.7, label='Regular Token')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Plot 2: Heatmap with token labels
    heatmap_data = np.array(steps).reshape(1, -1)
    im = ax2.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add token text and step labels
    for pos in range(min(gen_length, 64)):  # Limit to 64 for readability
        if pos < len(token_texts):
            token = token_texts[pos]
            step = steps[pos] if pos < len(steps) else -1

            # Truncate long tokens
            if len(token) > 8:
                token = token[:7] + '...'

            label = f"{token}\n[{step}]" if step >= 0 else token

            # Choose text color based on background
            text_color = 'white' if step > max(steps) / 2 else 'black'
            ax2.text(pos, 0, label, ha='center', va='center',
                    fontsize=7, color=text_color, fontweight='bold')

    ax2.set_xlabel('Position (Token / [Step])', fontsize=14, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_title('Heatmap: Token and Unmask Step', fontsize=16, fontweight='bold')
    ax2.set_xlim(-0.5, min(gen_length, 64) - 0.5)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Unmask Step', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")

    return fig


@torch.no_grad()
def generate_with_unmask_tracking(model, tokenizer, prompt, anchor_positions=None, anchor_token_ids=None,
                                   steps=128, gen_length=128, block_length=128,
                                   temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """
    Generate text and track the complete unmask order for all positions.

    Returns:
        x: generated sequence
        unmask_order: dict mapping position -> step_number (when it was unmasked)
        unmask_sequence: list of (step, positions_unmasked_at_this_step, tokens_revealed)
    """
    # Initialize tracking
    unmask_order = {}  # position -> step number
    unmask_sequence = []  # chronological list of unmask events

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Track anchor positions
    if anchor_positions is not None and anchor_token_ids is not None:
        for pos, token_id in zip(anchor_positions, anchor_token_ids):
            absolute_pos = prompt.shape[1] + pos
            if absolute_pos < x.shape[1]:
                x[0, absolute_pos] = token_id
                unmask_order[pos] = 0  # Anchors are unmasked at step 0

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

            # Track which positions are being unmasked at this step
            unmasked_positions = []
            unmasked_tokens = []

            for absolute_pos in range(prompt.shape[1], x.shape[1]):
                relative_pos = absolute_pos - prompt.shape[1]
                if transfer_index[0, absolute_pos]:
                    if relative_pos not in unmask_order:
                        unmask_order[relative_pos] = global_step + 1
                        unmasked_positions.append(relative_pos)
                        unmasked_tokens.append(x0[0, absolute_pos].item())

            if unmasked_positions:
                unmask_sequence.append({
                    'step': global_step + 1,
                    'positions': unmasked_positions,
                    'tokens': unmasked_tokens,
                    'num_unmasked': len(unmasked_positions)
                })

            x[transfer_index] = x0[transfer_index]
            global_step += 1

    return x, unmask_order, unmask_sequence


def main():
    device = 'cuda:2'

    print("Loading model and tokenizer...")
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "What is the capital of India?"
    m = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_text)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # Insert Tokyo as anchor
    anchor_text = " Tokyo"
    anchor_tokens = tokenizer.encode(anchor_text, add_special_tokens=False)
    anchor_positions = [5 + i for i in range(len(anchor_tokens))]
    anchor_token_ids = anchor_tokens

    print(f"\nPrompt: {prompt}")
    print(f"Inserting anchor '{anchor_text}' at positions {anchor_positions}")
    print(f"Anchor token IDs: {anchor_token_ids}")

    print("\nRunning generation with unmask order tracking...")
    output, unmask_order, unmask_sequence = generate_with_unmask_tracking(
        model, tokenizer, input_ids,
        anchor_positions=anchor_positions,
        anchor_token_ids=anchor_token_ids,
        steps=64, gen_length=64, block_length=64,
        temperature=0., cfg_scale=0., remasking='low_confidence'
    )

    final_output = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"Final output: {final_output}")

    # Create visualization
    print("\nCreating visualization...")
    create_unmask_visualization(
        unmask_order, tokenizer, output, input_ids.shape[1],
        anchor_positions=anchor_positions,
        save_path='/home/htseng23/LLaDA/unmask_order_visualization.png'
    )

    # Print summary table
    print("\n" + "="*80)
    print("UNMASK ORDER SUMMARY")
    print("="*80)
    print(f"{'Position':<10} {'Step':<10} {'Token':<30} {'Anchor?'}")
    print("-"*80)

    for pos in range(min(64, len(unmask_order))):
        if pos in unmask_order:
            step = unmask_order[pos]
            token_id = output[0, input_ids.shape[1] + pos].item()
            token_text = tokenizer.decode([token_id])
            is_anchor = "YES" if anchor_positions and pos in anchor_positions else ""
            print(f"{pos:<10} {step:<10} {repr(token_text):<30} {is_anchor}")

    print("="*80)


if __name__ == '__main__':
    main()
