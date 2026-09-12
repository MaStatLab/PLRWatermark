"""Wrong-key null: replay the PRF on the WATERMARKED text with an independent key.

Re-keying keeps temperature, prompts, model, text and repeated context/token
pairs fixed, making it a useful matched-text diagnostic. It does not remove
deterministic PRF reuse: a repeated pair yields the same pivot, so its later
conditional law is not a fresh uniform draw. An independent key alone does
not establish iid null paths or the conditional-null assumption for an anytime
guarantee. This module provides a helper, not a standalone analysis command.
"""
import sys, numpy as np

import real_data_experiment as real


def replay_with_key(prompts, tokens, *, vocabulary_size, key, horizon=200):
    torch = real._torch_module()
    p = torch.as_tensor(prompts, dtype=torch.long)
    t = torch.as_tensor(tokens, dtype=torch.long)
    ctx = torch.cat((p[:, -real.CONTEXT_WIDTH:], t[:, :horizon]), dim=1)[:, :horizon]
    table = real._official_fixed_hash_table(torch)
    seeds = table[(key * ctx) % real.PRF_TABLE_SIZE] + 1
    rows = int(t.shape[0])
    gum = np.empty((rows, horizon)); inv = np.empty_like(gum)
    gen = torch.Generator(device=torch.device("cpu")); den = vocabulary_size - 1
    for r in range(rows):
        for c in range(horizon):
            s = int(seeds[r, c].item()); tok = int(t[r, c].item())
            gen.manual_seed(s)
            gum[r, c] = float(torch.rand((vocabulary_size,), generator=gen)[tok].item())
            gen.manual_seed(s)
            u = torch.rand((1,), generator=gen)[0]
            rank = torch.randperm(vocabulary_size, generator=gen)[tok]
            inv[r, c] = abs(float(u.item()) - int(rank.item()) / den)
    return gum, inv
