"""Retrieval datasets for PRISM-Holo training.

The datasets most pertinent to waking the holographic read head, per the
paper §6.3: each (question, gold passage) pair teaches the encoder to map
semantically related text to similar bipolar vectors. Three sources covering
different retrieval regimes, plus a fluency backbone so the model doesn't
forget general language modeling while learning retrieval.

Dataset philosophy:
  * Natural Questions  — real user questions, Wikipedia answers (open-domain QA)
  * TriviaQA           — trivia fact recall (unambiguous key/value pairs)
  * MS-MARCO           — passage retrieval (the canonical IR benchmark)
  * FineWeb-Edu        — fluency backbone (prevent catastrophic forgetting)

Each retrieval example is encoded as a contiguous sequence::

    [BOS] question [SEP] answer/passage [EOS]

The training loss masks the question tokens (only supervise the answer), and
the retrieval-consistency loss (in holo_loss.py) operates on the seeded-tape
read signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from prism.train_scale import DatasetSpec


# ---------------------------------------------------------------------------
# Retrieval datasets (the core of Holo training)
# ---------------------------------------------------------------------------

MIX_HOLO_RETRIEVAL: list[DatasetSpec] = [
    DatasetSpec(
        # Natural Questions — real Google queries with Wikipedia gold passages.
        # The canonical open-domain QA benchmark; ideal for teaching the encoder
        # that "question about X" should map near "Wikipedia article about X".
        path="google-research-datasets/natural_questions",
        config="default",
        split="train",
        text_column=None,           # uses question + document columns
        weight=0.35,
        phase="pretrain",
    ),
    DatasetSpec(
        # TriviaQA — unambiguous fact pairs ("Who wrote X?" -> "Author").
        # Cleanest key/value signal; trains the binding operator directly.
        path="mandarjoshi/trivia_qa",
        config="rc.nocontext",
        split="train",
        text_column=None,
        weight=0.25,
        phase="pretrain",
    ),
    DatasetSpec(
        # MS-MARCO — passage retrieval. Teaches the encoder to handle longer
        # value texts (paragraphs, not just entity names).
        path="microsoft/ms_marco",
        config="v2.1",
        split="train",
        text_column=None,
        weight=0.20,
        phase="pretrain",
    ),
    DatasetSpec(
        # Fluency backbone — prevents catastrophic forgetting of general LM
        # while the encoder specializes for retrieval. Kept at 20% so the
        # model stays a general language model, not just a retriever.
        path="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        text_column="text",
        weight=0.20,
        phase="pretrain",
    ),
]


def get_holo_mix() -> list[DatasetSpec]:
    """Return the PRISM-Holo training mix."""
    return MIX_HOLO_RETRIEVAL


def holo_mix_summary(mix: list[DatasetSpec] | None = None) -> str:
    """Human-readable mix description."""
    mix = mix or MIX_HOLO_RETRIEVAL
    total = sum(d.weight for d in mix)
    lines = []
    for d in mix:
        pct = 100 * d.weight / total
        cfg = f"[{d.config}]" if d.config else ""
        kind = "retrieval" if d.text_column is None else "fluency"
        lines.append(f"  {pct:5.1f}%  {d.path}{cfg}  ({kind})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example formatting: turn a retrieval dataset row into a (text, mask) pair.
# ---------------------------------------------------------------------------


def format_retrieval_example(example: dict, tokenizer) -> tuple[str, str]:
    """Extract (question, answer) from a retrieval dataset row.

    Handles the schemas of NQ, TriviaQA, MS-MARCO. Returns the question text
    and the answer/passage text separately so the trainer can encode them with
    the HoloHead's key_encoder and value_encoder respectively.
    """
    # Natural Questions: question + annotations.short_answer or document text.
    if "question" in example:
        q = example["question"] if isinstance(example["question"], str) else example["question"].get("text", "")
        # NQ stores answers under annotations; fall back to long_answer candidates.
        ann = example.get("annotations") or {}
        a = (
            ann.get("short_answer") if isinstance(ann, dict) else None
            or _first_long_answer(example.get("long_answer_candidates"))
            or _first_short_answer(ann if isinstance(ann, dict) else {})
        )
        if not a and "document" in example:
            docs = example["document"]
            if isinstance(docs, dict):
                tokens = docs.get("tokens", {}).get("token", [])
                a = " ".join(tokens[:128]) if tokens else ""
        return q.strip(), (a or "").strip()

    # MS-MARCO: query + answers (list) or passages.
    if "query" in example:
        q = example["query"]
        answers = example.get("answers") or []
        if isinstance(answers, list) and answers:
            a = answers[0] if isinstance(answers[0], str) else str(answers[0])
        else:
            passages = example.get("passages", {}).get("passage_text", [])
            a = passages[0] if passages else ""
        return q.strip(), (a or "").strip()

    # TriviaQA: question + answer.value or answer.aliases.
    if "question" in example:
        return example["question"].strip(), str(example.get("answer", {}).get("value", "")).strip()

    # Fallback: concatenate any text-like fields.
    text_fields = [v for v in example.values() if isinstance(v, str) and len(v) > 20]
    if len(text_fields) >= 2:
        return text_fields[0].strip(), text_fields[1].strip()
    return "", ""


def _first_long_answer(candidates) -> str:
    if not candidates:
        return ""
    for c in candidates[:5]:
        if isinstance(c, dict):
            t = c.get("doc_token") or c.get("text")
            if t:
                return " ".join(t) if isinstance(t, list) else str(t)
    return ""


def _first_short_answer(ann: dict) -> str:
    sa = ann.get("short_answer") or ann.get("short_answers")
    if not sa:
        return ""
    if isinstance(sa, list) and sa:
        first = sa[0]
        if isinstance(first, dict):
            return first.get("text", str(first))
        return str(first)
    return str(sa)
