#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entrypoint restricted to canonical pretraining kernels."""

from __future__ import annotations

import sys

from kernel_benchmark import main


TRAINING_IDENTITIES = (
    "rwkv-lm/pretrain_recurrent_fp32io16_forward",
    "rwkv-lm/pretrain_recurrent_fp32io16_backward",
    "fla/pretrain_chunk_fp32io16_forward",
    "fla/pretrain_chunk_fp32io16_backward",
)


if __name__ == "__main__":
    if "--identities" not in sys.argv:
        sys.argv.extend(("--identities", *TRAINING_IDENTITIES))
    main()
