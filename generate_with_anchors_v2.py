import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def visualize_sequence(tokenizer, sequence, mask_id=126336):
    """
    Visualize a sequence showing [MASK] tokens and actual tokens.
    """
    tokens = []
    for token_id in sequence[0].cpu().numpy():
        if token_id == mask_id:
            tokens.append("[MASK]")
        else:
            # Decode individual token
            token_text = tokenizer.decode([token_id])
            tokens.append(f"'{token_text}'")
    return " ".join(tokens)


def print_output_tokens(tokenizer, output_sequence, verbose=True):
    """
    Print the output tokens with their IDs, including special tokens.
    """
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("OUTPUT TOKEN ANALYSIS")
    print("=" * 80)

    # Get token IDs
    token_ids = output_sequence[0].cpu().numpy()

    print(f"Total tokens: {len(token_ids)}")
    print("\nToken-by-token breakdown:")
    print(f"{'Index':<6} {'Token ID':<10} {'Token Text':<30} {'Special?'}")
    print("-" * 80)

    for idx, token_id in enumerate(token_ids):
        # Decode the token
        token_text = tokenizer.decode([token_id])

        # Check if it's a special token
        is_special = token_id in tokenizer.all_special_ids
        special_name = ""

        if is_special:
            # Try to identify which special token it is
            if token_id == tokenizer.eos_token_id:
                special_name = " (EOS/STOP)"
            elif token_id == tokenizer.bos_token_id:
                special_name = " (BOS)"
            elif token_id == tokenizer.pad_token_id:
                special_name = " (PAD)"
            else:
                special_name = " (SPECIAL)"

        # Truncate long tokens for display
        display_text = repr(token_text)[:30]

        print(f"{idx:<6} {token_id:<10} {display_text:<30} {special_name}")

    print("=" * 80 + "\n")


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate_with_anchors(model, tokenizer, prompt, anchor_positions=None, anchor_token_ids=None,
                          steps=128, gen_length=128, block_length=128, temperature=0.,
                          cfg_scale=0., remasking='low_confidence', mask_id=126336, verbose=True,
                          track_positions=None, top_k=5):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        anchor_positions: List of positions (relative to start of generation) to place anchor tokens.
                         E.g., [5, 10, 15] means positions 5, 10, 15 in the generated sequence.
        anchor_token_ids: List of token IDs to place at the anchor positions.
                         Must be same length as anchor_positions.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        track_positions: List of positions (relative to start of generation) to track probability distributions.
                        E.g., [5, 10] will track the top-k tokens at positions 5 and 10 across all steps.
        top_k: Number of top tokens to display (default 5).
    '''

    # Create fully masked sequence
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Insert anchor tokens at specified positions
    if anchor_positions is not None and anchor_token_ids is not None:
        assert len(anchor_positions) == len(anchor_token_ids), "anchor_positions and anchor_token_ids must have same length"
        for pos, token_id in zip(anchor_positions, anchor_token_ids):
            # Position is relative to the start of generated sequence
            absolute_pos = prompt.shape[1] + pos
            if absolute_pos < x.shape[1]:
                x[0, absolute_pos] = token_id
                if verbose:
                    print(f"Inserted anchor token {token_id} at position {pos} (absolute: {absolute_pos})")

    # Visualize the anchor-inserted sequence
    if verbose and anchor_positions is not None:
        print("\nInitial sequence with anchors (generation part only):")
        gen_part = x[:, prompt.shape[1]:]
        viz = visualize_sequence(tokenizer, gen_part, mask_id)
        print(viz[:500])  # Limit output length
        if len(viz) > 500:
            print("... (truncated)")
        print()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # Print header for tracking if enabled
    if track_positions is not None and verbose:
        print("\n" + "=" * 80)
        print("TRACKING PROBABILITY DISTRIBUTIONS AT SPECIFIED POSITIONS")
        print("=" * 80)

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
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            # Calculate probabilities
            p = F.softmax(logits, dim=-1)

            # Track and print top-k tokens at specified positions
            if track_positions is not None and verbose:
                global_step = num_block * steps_per_block + i
                for track_pos in track_positions:
                    absolute_track_pos = prompt.shape[1] + track_pos
                    if absolute_track_pos < x.shape[1]:
                        # Get probabilities for this position
                        pos_probs = p[0, absolute_track_pos, :]
                        # Get top-k tokens and their probabilities
                        top_probs, top_indices = torch.topk(pos_probs, k=min(top_k, pos_probs.shape[0]))

                        # Get the predicted token at this position
                        predicted_token = x0[0, absolute_track_pos].item()
                        is_masked = mask_index[0, absolute_track_pos].item()

                        print(f"\nStep {global_step:3d} | Position {track_pos:3d} | Masked: {is_masked}")
                        if i < 20:
                            print(f"  Top {top_k} tokens:")
                            for rank, (token_id, prob) in enumerate(zip(top_indices.cpu().tolist(), top_probs.cpu().tolist()), 1):
                                token_text = tokenizer.decode([token_id])
                                marker = " <- PREDICTED" if token_id == predicted_token else ""
                                print(f"    {rank}. {token_text:20s} (ID: {token_id:6d}, prob: {prob:.4f}){marker}")

            if remasking == 'low_confidence':
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
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

    if track_positions is not None and verbose:
        print("\n" + "=" * 80)
        print("END OF TRACKING")
        print("=" * 80 + "\n")

    return x


def main():
    device = 'cuda:2'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # Example question
    prompt = "In what year was Jane Austen born?"
    m = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_text)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # Define your anchors as (position, text) pairs
    # IMPORTANT: If text tokenizes to multiple tokens, they will be inserted sequentially
    anchors = [
         (6, "2022"),
    ]

    # Convert to the format needed by the function
    # This properly handles multi-token anchors
    anchor_positions = []
    anchor_token_ids = []

    for pos, text in anchors:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"Anchor '{text}' tokenizes to {len(tokens)} tokens: {tokens}")
        print(f"  Token texts: {[tokenizer.decode([t]) for t in tokens]}")

        # Insert all tokens sequentially starting from pos
        for i, token_id in enumerate(tokens):
            anchor_positions.append(pos + i)
            anchor_token_ids.append(token_id)

    print(f"\nFinal anchor positions: {anchor_positions}")
    print(f"Final anchor token IDs: {anchor_token_ids}")

    track_positions = [2, 4, 5, 6, 7]  # Track around the anchor

    out_anchored = generate_with_anchors(model, tokenizer, input_ids,
                                         anchor_positions=anchor_positions,
                                         anchor_token_ids=anchor_token_ids,
                                         steps=64, gen_length=64,
                                         block_length=64, temperature=0.,
                                         cfg_scale=0., remasking='low_confidence',
                                         verbose=True,
                                         track_positions=track_positions,
                                         top_k=5)
    result_anchored = tokenizer.batch_decode(out_anchored[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"Final Output: {result_anchored}")

    # Print token analysis for anchored output
    print_output_tokens(tokenizer, out_anchored[:, input_ids.shape[1]:], verbose=True)


if __name__ == '__main__':
    main()
