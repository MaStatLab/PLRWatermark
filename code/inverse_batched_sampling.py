#!/usr/bin/env python3
"""Batch-correct inverse-transform generation, equivalent to the released code.

The released ``transform_key_func`` refuses any batch larger than one::

    batch_size = inputs.shape[0] # batch_size must be 1
    assert batch_size == 1, "Batch size should be 1 for the inverse transform watermark!"
    for _ in range(batch_size):
        seed_rng(generator, inputs, ...)
        xi = torch.rand(size=(batch_size,1), generator=generator)

The assertion guards a real defect rather than an inherent constraint: the loop
seeds on the *whole* ``inputs`` tensor on every pass instead of on row ``k``, and
shapes ``xi`` as ``(batch_size, 1)`` inside the loop, so at any batch above one
it would seed identically for every row and stack the wrong shapes.  The
released Gumbel path, ``gumbel_key_func``, already does it correctly with
``for k in range(inputs.shape[0])`` and ``seed_rng(..., inputs[k].unsqueeze(0), ...)``.
This module applies that same pattern to the inverse path, and to the sampler
and pivot function, which are likewise written for a single row.

Nothing here changes the watermark.  Every function reduces to the released one
at batch size one, which :mod:`test_inverse_batched_sampling` asserts element by
element against the upstream implementations, including the order in which
random draws are taken from the generator.

Batching matters only because it makes the experiment affordable: the per-row
key work is a ``torch.rand`` and a ``torch.randperm``, about 0.18 ms at the
OPT-1.3B vocabulary, so 500 documents of 200 tokens cost roughly 18 s of key
construction.  The model forward passes dominate, and those batch normally.
"""

from __future__ import annotations

import torch


def transform_key_func_batched(
    generator, inputs, vocab_size, key, c, seeding_scheme
):
    """Per-row inverse-transform key, mirroring ``gumbel_key_func``'s loop.

    Draw order within a row is ``rand`` then ``randperm``, matching the released
    ``transform_key_func`` exactly so a batch of one reproduces it bit for bit.
    """
    from sampling import seed_rng

    xis = []
    pis = []
    for k in range(inputs.shape[0]):
        # seed_rng documents that it wants (1, length); the released inverse
        # path passes the full batch here, which is the defect being fixed.
        seed_rng(
            generator,
            inputs[k].unsqueeze(0),
            seeding_scheme=seeding_scheme,
            hash_key=key,
            c=c,
        )
        xis.append(torch.rand(size=(1, 1), generator=generator))
        pis.append(torch.randperm(vocab_size, generator=generator).unsqueeze(0))
    return torch.vstack(xis), torch.vstack(pis)


def inverse_permutation_batched(pi: torch.Tensor) -> torch.Tensor:
    """Row-wise inverse of a (batch, vocabulary) permutation tensor.

    The released ``inverse_permutation`` assigns ``inv[perm] = arange(V)`` for a
    single row; scattering along dimension one is the same statement per row.
    """
    positions = torch.arange(pi.shape[1], device=pi.device).expand_as(pi)
    return torch.empty_like(pi).scatter_(1, pi, positions)


def transform_sampling_batched(probs, pi, xi):
    """Released ``transform_sampling`` without the single-row squeeze."""
    inv_pi = inverse_permutation_batched(pi)
    cdf = torch.cumsum(torch.gather(probs, 1, inv_pi), 1)
    return torch.gather(inv_pi, 1, torch.searchsorted(cdf, xi))


def transform_Y_batched(s, pi, xi):
    """Released ``transform_Y`` with the batch dimension kept throughout.

    Returns ``(Y, U, eta)`` shaped ``(batch, 1)``.  The released version squeezes
    ``s_samp`` to a scalar, which is only correct for one row; keeping the column
    avoids the ``(batch, 1) - (batch,)`` broadcast that would otherwise produce a
    ``(batch, batch)`` outer difference.  The sign convention is theirs:
    ``Y = -|U - eta|`` so that ``E_0 Y < E_1 Y``.
    """
    vocab_size = pi.shape[1]
    s_samp = torch.gather(pi, -1, s.cpu())
    eta = (s_samp - 1) / (vocab_size - 1)
    return -torch.abs(xi - eta), xi, eta


def generate_inv_batched(
    model, prompts, vocab_size, m, *, key=23333, c=5,
    seeding_scheme="minhash_prf", temperature=0.1,
):
    """Released ``generate_inv`` loop, batched.

    Mirrors the upstream body statement for statement -- same KV-cache reuse,
    same attention-mask growth, same division of the logits by ``temperature``
    -- substituting the batch-correct key, sampler and pivot functions.
    """
    generator = torch.Generator()
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None

    pivots, uniforms, etas, top_probs = [], [], [], []
    for _ in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(
            output.logits[:, -1] / temperature, dim=-1
        ).cpu()
        top_probs.append(torch.max(probs, axis=1)[0].unsqueeze(1))

        xi, pi = transform_key_func_batched(
            generator, inputs, vocab_size, key, c, seeding_scheme
        )
        tokens = transform_sampling_batched(probs, pi, xi).to(model.device)
        pivot, uniform, eta = transform_Y_batched(tokens, pi, xi)

        inputs = torch.cat([inputs, tokens], dim=-1)
        pivots.append(pivot)
        uniforms.append(uniform)
        etas.append(eta)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    def stack(chunks):
        return torch.cat(chunks, dim=1).detach().cpu()

    return (
        inputs.detach().cpu(),
        stack(pivots), stack(uniforms), stack(etas), stack(top_probs),
    )
