"""OneShotLearner — zero-gradient in-context learning via the memory tape.

PRISM-KB Module 4. Add (input, output) example pairs; at generation time the
lesson is encoded into slots and seeded into the tape via
``MemoryState.from_knowledge``. The MemoryExpert's read head then retrieves the
relevant slot when the query matches an input, and the decoder turns the
retrieved slot into the output.

"Learning" here is literally zero backward passes — it's an encode + a tensor
concatenation. Whether retrieval actually fires correctly depends on the model
having been trained to read seeded content (Phase 2 of the proposal); this
module implements the mechanism, and the honest validation test measures whether
it fires.

Supports a FewShot variant (many examples) for when the single-shot capacity
(num_slots) is exhausted.
"""

from __future__ import annotations

import torch
from torch import nn

from prism.config import PrismConfig
from prism.memory import MemoryState


class OneShotLearner:
    """Zero-gradient one-shot / few-shot learner over the memory tape.

    Args:
        model: a trained Prism model.
        encoder: a PrismEncoder producing slots from text.
        config: the PrismConfig (for memory dims).
    """

    def __init__(self, model: nn.Module, encoder, config: PrismConfig) -> None:
        self.model = model
        self.encoder = encoder
        self.config = config
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self._lesson: list[tuple[str, str]] = []

    def add_example(self, inp: str, out: str) -> None:
        """Add a single (input, output) pair to the lesson."""
        self._lesson.append((inp, out))

    def clear(self) -> None:
        self._lesson.clear()

    @property
    def num_examples(self) -> int:
        return len(self._lesson)

    def _build_lesson_slots(self, tokenizer) -> torch.Tensor:
        """Encode all lesson pairs into a flat (K, d_mem) tensor.

        Each pair is encoded as ``f"{inp} -> {out}"`` so the read head sees the
        input and output together in the same slot(s).
        """
        if not self._lesson:
            return torch.zeros(0, self.config.memory.d_mem, device=self.device, dtype=self.dtype)
        texts = [f"{inp} -> {out}" for inp, out in self._lesson]
        slots = self.encoder.encode_texts(texts, tokenizer, self.device)   # (N, k, d_mem)
        return slots.reshape(-1, self.config.memory.d_mem)                 # (N*k, d_mem)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        blend_ratio: float = 1.0,
    ) -> str:
        """Generate text from ``prompt``, seeded with the current lesson.

        No backward pass. The lesson slots seed the tape; the tape is carried
        forward across generation steps (final_mem).
        """
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        lesson_slots = self._build_lesson_slots(tokenizer)
        mem = MemoryState.from_knowledge(
            kb_slots=lesson_slots,
            batch_size=1,
            config=self.config.memory,
            device=self.device,
            dtype=self.dtype,
            blend_ratio=blend_ratio,
        )

        generated = []
        eos = getattr(tokenizer, "eos_token_id", None)
        for _ in range(max_new_tokens):
            out = self.model(input_ids, mem=mem)
            logits = out.logits[:, -1, :]
            mem = out.final_mem   # carry the tape forward
            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            tid = next_id.item()
            generated.append(tid)
            if eos is not None and tid == eos:
                break
        return tokenizer.decode(torch.tensor(generated).unsqueeze(0)[0], skip_special_tokens=True)


class FewShotLearner(OneShotLearner):
    """Same mechanism, explicit naming for the multi-example case.

    Capacity is bounded by ``config.memory.num_slots // num_slots_per_doc``.
    Beyond that, retrieve_per_step (in generate.py) re-selects slots.
    """
