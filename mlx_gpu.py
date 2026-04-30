"""Process-wide MLX/Metal serialization.

The embedding model (sfr-embed / qodo) and the description model (gemma-3-4b)
both run on the same Metal GPU. When dispatched from concurrent threads — e.g.
the watcher's initial-scan thread generating descriptions while a search
request embeds a query — they hit a Metal command-buffer collision:

    -[AGXG14XFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:]:1094:
    failed assertion `A command encoder is already encoding to this command buffer'

This crashes the process. The fix is to make every MLX call site take this
re-entrant lock so only one thread is dispatching to Metal at a time.

Use it as a context manager:

    from mlx_gpu import GPU
    with GPU:
        output = mlx_generate(model, tokenizer, texts=texts)

`RLock` (re-entrant) so a function that holds the lock can safely call into
another helper that also takes it without deadlocking.
"""

import threading

GPU = threading.RLock()
