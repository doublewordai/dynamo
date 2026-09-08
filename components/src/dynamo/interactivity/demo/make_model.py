# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a tiny local tokenizer; no downloaded model or weights."""

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

path = Path("/opt/model")
path.mkdir(parents=True, exist_ok=True)
tokenizer = Tokenizer(
    models.WordLevel({"[UNK]": 0, "[EOS]": 1, "[BOS]": 2, "demo": 3}, unk_token="[UNK]")
)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=False)
tokenizer.save(str(path / "tokenizer.json"))
(path / "config.json").write_text(
    json.dumps(
        dict(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
            vocab_size=4,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=4096,
            bos_token_id=2,
            eos_token_id=1,
            torch_dtype="float32",
        )
    )
)
(path / "tokenizer_config.json").write_text(
    json.dumps(
        dict(
            tokenizer_class="PreTrainedTokenizerFast",
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            model_max_length=4096,
            chat_template="{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant: ",
        )
    )
)
