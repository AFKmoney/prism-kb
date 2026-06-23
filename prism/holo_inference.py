"""HoloInference — end-to-end knowledge-driven generation.

The user-facing API for PRISM-Holo at inference: bind facts into the tape,
then generate text conditioned on the seeded tape. This is Axis 2 in action
(zero gradient — facts are added algebraically, the model answers using them).

Two paths:

  * ``generate_with_facts(model, encoder, prompt, facts, tokenizer)``: bind
    a list of (key, value) facts into a HoloTape, seed the model's memory from
    it, and generate.
  * ``answer_with_kb(model, encoder, kb, prompt, tokenizer)``: retrieve the
    top-k relevant slots from a persisted KnowledgeBase, seed, generate.

Both compose with the PRISM-Holo architecture (holo_mode=True). They DO NOT
modify model weights — the tape is a runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from prism.config import PrismConfig
from prism.holo import HoloEncoder, HoloTape
from prism.kb import KnowledgeBase
from prism.memory import MemoryState


@dataclass
class HoloAnswer:
    """Result of a HoloInference call.

    Attributes:
        text: generated text.
        retrieved_sim: mean cosine similarity of the retrieved slots to the
            query (high = relevant facts found).
        tape_summary: state of the HoloTape after binding (capacity used).
    """

    text: str
    retrieved_sim: float
    tape_summary: str


class HoloInference:
    """Knowledge-driven generation for PRISM-Holo models.

    Args:
        model: a Prism model (holo_mode=True recommended).
        encoder: a HoloEncoder matching the model's d_model and D.
        config: the PrismConfig.
        tokenizer: a HF tokenizer.
    """

    def __init__(self, model: nn.Module, encoder: HoloEncoder, config: PrismConfig, tokenizer) -> None:
        self.model = model
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.D = config.memory.num_slots * config.memory.d_mem
        model.eval()

    def _seed_from_tape(self, tape: HoloTape) -> MemoryState:
        """Build a MemoryState by seeding the model's tape from a HoloTape.

        The HoloTape is a flat (D,) register; we reshape it to (num_slots, d_mem)
        to fit the existing MemoryState shape.
        """
        H = tape.H.reshape(self.config.memory.num_slots, self.config.memory.d_mem)
        return MemoryState.from_knowledge(
            H.unsqueeze(0), batch_size=1, config=self.config.memory,
            device=self.device, dtype=self.dtype,
        )

    @torch.no_grad()
    def generate_with_facts(
        self,
        prompt: str,
        facts: list[tuple[str, str]],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> HoloAnswer:
        """Bind facts into a fresh HoloTape, seed the model, generate.

        Args:
            prompt: the question to answer.
            facts: list of (key, value) text pairs to bind.
            max_new_tokens: generation cap.
            temperature: 0 = greedy, >0 = sampled.

        Returns:
            HoloAnswer with the generated text and retrieval diagnostics.
        """
        tape = HoloTape(D=self.D)
        for key_text, val_text in facts:
            k_emb = self._encode_text(key_text)
            v_emb = self._encode_text(val_text)
            tape.bind(self.encoder(k_emb), self.encoder(v_emb))

        mem = self._seed_from_tape(tape)
        text = self._generate(prompt, mem, max_new_tokens, temperature)

        # Diagnostics: how well does the prompt retrieve the bound facts?
        q_emb = self._encode_text(prompt)
        q_holo = self.encoder(q_emb)
        retrieved = tape.unbind(q_holo)
        sims = []
        for _, val_text in facts:
            v_emb = self._encode_text(val_text)
            v_holo = self.encoder(v_emb)
            sims.append(F.cosine_similarity(retrieved.unsqueeze(0), v_holo.unsqueeze(0)).item())
        return HoloAnswer(
            text=text,
            retrieved_sim=sum(sims) / max(len(sims), 1),
            tape_summary=tape.summary(),
        )

    @torch.no_grad()
    def answer_with_kb(
        self,
        kb: KnowledgeBase,
        prompt: str,
        top_k: int = 8,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> HoloAnswer:
        """Retrieve top-k slots from a KB, seed, generate.

        Args:
            kb: a persisted KnowledgeBase.
            prompt: the question.
            top_k: number of slots to retrieve.

        Returns:
            HoloAnswer.
        """
        q_emb = self._encode_text(prompt)
        # Retrieve from the KB. The KB stores in d_mem space; for the demo we
        # treat the encoder output's first d_mem dims as the query for retrieval.
        q_for_retrieval = self.encoder(q_emb)[: self.config.memory.d_mem]
        retrieved_slots = kb.retrieve(q_for_retrieval, top_k=top_k).to(self.device, self.dtype)

        mem = MemoryState.from_knowledge(
            retrieved_slots, batch_size=1, config=self.config.memory,
            device=self.device, dtype=self.dtype,
        )
        text = self._generate(prompt, mem, max_new_tokens, temperature)
        return HoloAnswer(
            text=text,
            retrieved_sim=0.0,  # not measured for KB path (different space)
            tape_summary=f"KB: {len(kb)} docs, retrieved {top_k} slots",
        )

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode a text string to a (d_model,) embedding via the model."""
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = enc["input_ids"].to(self.device)
        # Mean-pool the model's token embeddings (no gradient).
        with torch.no_grad():
            x = self.model.embed(input_ids).squeeze(0).mean(dim=0)
        return x

    def _generate(self, prompt: str, mem: MemoryState, max_new_tokens: int, temperature: float) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = enc["input_ids"].to(self.device)
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
        return self.tokenizer.decode(torch.tensor(generated), skip_special_tokens=True)


def description() -> str:
    return (
        "HoloInference: bind facts into the holographic tape, then generate. "
        "Axis 2 in action — knowledge added with zero gradient, model answers "
        "using it. Two paths: explicit facts list or persisted KnowledgeBase."
    )
