"""CogLoop — the full cognitive loop tying everything together.

PERCEIVE → REFLECT → RESPOND → CONSOLIDATE.

    ┌──────────────────────────────────────────────────────────┐
    │                       COGLOOP                            │
    │                                                          │
    │  question ──▶ PERCEIVE ──▶ REFLECT ──▶ RESPOND ──▶ CONSOLIDATE
    │                │             │           │             │
    │            capture       multi-pass    generate     observe +
    │            + retrieve    on memory     seeded by    maybe
    │                                          ctx_N       persist
    └──────────────────────────────────────────────────────────┘

This is the user-facing API: instantiate CogLoop, call .answer(question),
optionally .remember(...). Memory persists across calls (and across sessions
via the LongTermStore on disk).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from prism.capture import AnalyticCapture
from prism.cogmemory import CogMemory, Episode
from prism.config import PrismConfig
from prism.kb import KnowledgeBase
from prism.reflect import ReflectConfig, Reflector


@dataclass
class CogAnswer:
    """The result of one CogLoop.answer() call."""

    text: str
    passes_used: int
    converged: bool
    consolidated: bool
    memory_summary: str


class CogLoop:
    """The full cognitive loop. One instance per PRISM model + memory store.

    Args:
        model: a Prism model.
        config: the PrismConfig.
        tokenizer: a HF tokenizer.
        long_term_path: where the persistent KB lives on disk.
        use_capture: if True, PERCEIVE uses AnalyticCapture (zero-GPU alignment).
            If False, skips capture (reflection runs on an empty/seeded tape).
        reflect_config: reflection hyperparameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PrismConfig,
        tokenizer,
        long_term_path: str,
        use_capture: bool = True,
        reflect_config: ReflectConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self.memory = CogMemory(d_mem=config.memory.d_mem, long_term_path=long_term_path)
        self.capture = AnalyticCapture(model, config) if use_capture else None
        self.reflector = Reflector(
            model, config,
            kb=self.memory.long_term.kb,
            reflect_config=reflect_config,
            encoder=self.capture,   # Reflector uses it for retrieval queries
        )
        self.model.eval()

    @torch.no_grad()
    def answer(
        self,
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        importance: float = 0.0,
    ) -> CogAnswer:
        """Answer a question through the full cognitive loop.

        Args:
            question: the user's question.
            max_new_tokens: generation cap.
            temperature: 0.0 = greedy, >0 = sampled.
            importance: salience to assign this episode (affects consolidation).

        Returns:
            CogAnswer with the response + diagnostics.
        """
        # --- PERCEIVE: encode the question, optionally capture its slots. ---
        enc = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc["input_ids"].to(self.device)
        seed_slots = None
        if self.capture is not None:
            seed_slots = self.capture.capture_ids(input_ids).slots

        # --- REFLECT: multi-pass retrieval + reasoning on memory. ---
        ctx, trace = self.reflector.reflect(input_ids, initial_seed_slots=seed_slots)

        # --- RESPOND: generate seeded by the reflected context. ---
        from prism.memory import MemoryState

        mem = MemoryState.from_knowledge(
            ctx, batch_size=1, config=self.config.memory,
            device=self.device, dtype=self.dtype,
        )
        answer_ids = self._generate(input_ids, mem, max_new_tokens, temperature)
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        # --- CONSOLIDATE: observe this episode; maybe persist. ---
        episode = Episode(
            text=f"Q: {question}\nA: {answer_text}",
            slots=ctx.detach().cpu().tolist() if ctx.numel() > 0 else [],
            importance=importance,
            metadata={"source": "answer"},
        )
        self.memory.observe(episode)

        return CogAnswer(
            text=answer_text,
            passes_used=trace.passes_used,
            converged=trace.converged,
            consolidated=len(self.memory.long_term) > 0,
            memory_summary=self.memory.summary(),
        )

    def remember(self, text: str, importance: float = 1.0) -> None:
        """Explicitly commit a fact to long-term memory (immediate consolidation).

        Captures the text's slots via AnalyticCapture so the stored slots are in
        the read head's native distribution.
        """
        if self.capture is None:
            slots = torch.zeros(self.config.memory.num_slots, self.config.memory.d_mem)
        else:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            slots = self.capture.capture_ids(enc["input_ids"].to(self.device)).slots
        self.memory.remember_explicitly(text, slots, importance=importance)

    def _generate(self, input_ids, mem, max_new_tokens, temperature):
        """Greedy or sampled generation, carrying the tape forward."""
        generated = []
        eos = self.tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            out = self.model(input_ids, mem=mem)
            logits = out.logits[:, -1, :]
            mem = out.final_mem
            if temperature == 0.0:
                nid = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                nid = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, nid], dim=1)
            generated.append(nid.item())
            if eos is not None and nid.item() == eos:
                break
        return torch.tensor(generated)


def description() -> str:
    return (
        "CogLoop: PERCEIVE -> REFLECT -> RESPOND -> CONSOLIDATE. The full "
        "cognitive loop. PRISM 'thinks' (multi-pass reflection on persistent "
        "memory) before answering, and consolidates salient episodes to "
        "long-term storage like a human."
    )
