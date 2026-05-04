import json
import re

import httpx
from fastapi import HTTPException


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

AGENT_PLAN_PROMPT = """You are an academic research assistant embedded in a PDF reader.
This is turn 1 of a bounded highlight-finding loop. You MUST NOT produce highlights yet.
Create a clear, compact plan for finding non-duplicate passages that answer the user's question.

Respond ONLY with valid JSON:
{
  "plan": "2-4 sentence plan",
  "categories": [
    {"label": "Short label", "color": "#hexcolor", "purpose": "what this category captures"}
  ],
  "target_pages": [1, 2],
  "keywords": ["term"],
  "selection_strategy": "how later turns should avoid duplicates"
}

Rules:
- 2-5 categories max.
- Pick readable, distinct colors.
- target_pages may be empty if the index is not enough.
- No markdown, no commentary outside JSON."""

AGENT_CANDIDATE_PROMPT = """You are an academic research assistant embedded in a PDF reader.
This is a middle turn in a bounded highlight-finding loop. Select candidate highlights only from the provided excerpts.

Respond ONLY with valid JSON:
{
  "candidates": [
    {
      "block_id": "B001",
      "page": 1,
      "text": "Exact verbatim substring from the provided block",
      "label": "One category label",
      "color": "#hexcolor"
    }
  ]
}

Rules:
- The text MUST be an exact substring from the provided block. Copy it verbatim.
- Do NOT select text that duplicates or substantially overlaps any already_selected item.
- Do NOT select more than one highlight from the same paragraph/block unless the user explicitly asked for line-by-line coverage.
- Prefer compact passages that make sense as standalone highlights.
- If no new useful highlights exist in the excerpts, return {"candidates":[]}.
- No markdown, no commentary outside JSON."""

AGENT_FINAL_PROMPT = """You are an academic research assistant embedded in a PDF reader.
This is the final turn of a bounded highlight-finding loop. Build the final highlight JSON from the accumulated candidates.

Respond ONLY with valid JSON:
{
  "legend": [
    {"label": "Short label", "color": "#hexcolor"}
  ],
  "highlights": [
    {
      "text": "Exact verbatim substring from a candidate",
      "page": 1,
      "color": "#hexcolor"
    }
  ]
}

Rules:
- Use only candidate text provided to you. Do not invent new passages.
- Remove duplicates and near-duplicates.
- Keep at most one highlight per paragraph/block.
- Include only highlights that genuinely answer the user's question.
- Colors must match legend entries.
- If nothing relevant exists, return empty arrays.
- No answer field, no commentary, no markdown."""


def strip_json_fence(content: str) -> str:
    content = (content or "").strip()
    if content.startswith("```"):
        lines = content.split("\n")
        return "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    return content


def parse_json_content(content: str) -> dict:
    content = strip_json_fence(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    starts = [idx for idx in (content.find("{"), content.find("[")) if idx >= 0]
    if not starts:
        raise json.JSONDecodeError("No JSON object found", content, 0)
    start = min(starts)
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(content[start:])
    return parsed


async def call_openrouter_json(
    client: httpx.AsyncClient,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4000,
    temperature: float = 0.25,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "MorePapersPlease",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(3):
        response = await client.post(OPENROUTER_URL, json=payload, headers=headers)

        if response.status_code != 200:
            detail = response.text[:500]
            raise HTTPException(502, f"OpenRouter error ({response.status_code}): {detail}")

        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise HTTPException(502, f"OpenRouter: {msg}")

        choices = data.get("choices") or []
        if not choices:
            raise HTTPException(502, f"OpenRouter returned no choices. Raw: {json.dumps(data)[:400]}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = message.get("reasoning_content") or message.get("reasoning") or ""
        content = strip_json_fence(content)
        if not content:
            finish = choices[0].get("finish_reason", "?")
            usage = data.get("usage", {})
            raise HTTPException(
                502,
                f"Model '{model}' returned empty content (finish_reason={finish}, usage={usage}). "
                f"Possible causes: rate limit, no credits, content policy block, output truncation, "
                f"or the model genuinely returned nothing. Check OpenRouter dashboard.",
            )

        try:
            parsed = parse_json_content(content)
            return parsed if isinstance(parsed, dict) else {"items": parsed}
        except json.JSONDecodeError:
            if attempt == 2:
                raise HTTPException(500, "AI returned invalid JSON")

    raise HTTPException(500, "AI returned invalid JSON")


def build_blocks(extracted: list) -> list[dict]:
    blocks = []
    idx = 1
    for page_data in extracted:
        for block in page_data.get("blocks", []):
            text = (block.get("text") or "").strip()
            if not text:
                continue
            blocks.append({
                "id": f"B{idx:03d}",
                "page": page_data["page"],
                "text": text,
                "bbox": block.get("bbox"),
            })
            idx += 1
    return blocks


def compact_paper_index(blocks: list[dict], max_chars: int = 320) -> str:
    lines = []
    for block in blocks:
        snippet = re.sub(r"\s+", " ", block["text"])[:max_chars]
        lines.append(f"{block['id']} | page {block['page']} | {snippet}")
    return "\n".join(lines)


def block_excerpt(block: dict, max_chars: int = 2200) -> str:
    text = block["text"]
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def find_text_in_blocks(blocks: list[dict], search_text: str, page_hint: int | None = None) -> dict | None:
    search_clean = (search_text or "").strip().lower()
    if not search_clean:
        return None

    candidates = blocks
    if page_hint:
        page_candidates = [block for block in blocks if block["page"] == page_hint]
        if page_candidates:
            candidates = page_candidates

    search_start = search_clean[:80]
    for block in candidates:
        if search_start in block["text"].lower():
            return block

    words = [word for word in re.split(r"\W+", search_clean)[:6] if len(word) > 4]
    for block in candidates:
        block_text = block["text"].lower()
        if words and sum(1 for word in words if word in block_text) >= min(3, len(words)):
            return block

    return None


def dedupe_candidates(candidates: list[dict], blocks: list[dict]) -> list[dict]:
    seen_texts = set()
    seen_blocks = set()
    deduped = []
    for item in candidates:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        page = item.get("page")
        block = None
        block_id = item.get("block_id")
        if block_id:
            block = next((b for b in blocks if b["id"] == block_id), None)
        if block is None:
            block = find_text_in_blocks(blocks, text, page)
        norm = normalize_text(text)
        if not norm:
            continue
        block_key = block["id"] if block else f"page-{page}:{norm[:80]}"
        if norm in seen_texts or block_key in seen_blocks:
            continue
        seen_texts.add(norm)
        seen_blocks.add(block_key)
        deduped.append({
            "block_id": block_key,
            "page": block["page"] if block else page,
            "text": text,
            "label": item.get("label", ""),
            "color": item.get("color", "#ffeb3b"),
        })
    return deduped


def rank_blocks(blocks: list[dict], plan: dict) -> list[dict]:
    pages = {int(p) for p in plan.get("target_pages", []) if str(p).isdigit()}
    keywords = [str(k).lower() for k in plan.get("keywords", []) if str(k).strip()]
    scored = []
    for i, block in enumerate(blocks):
        text = block["text"].lower()
        score = 0
        if block["page"] in pages:
            score += 8
        score += sum(2 for keyword in keywords if keyword in text)
        if score == 0 and not pages and not keywords:
            score = 1
        scored.append((score, i, block))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [block for score, _, block in scored if score > 0]
    return ranked or blocks


def make_candidate_user_message(
    question: str,
    plan: dict,
    excerpts: list[dict],
    already_selected: list[dict],
    pass_number: int,
) -> str:
    excerpt_text = "\n\n".join(
        f"BLOCK {block['id']} PAGE {block['page']}:\n{block_excerpt(block)}"
        for block in excerpts
    ) or "(No excerpts in this pass.)"
    already = [
        {
            "block_id": item.get("block_id"),
            "page": item.get("page"),
            "text": item.get("text"),
            "label": item.get("label"),
        }
        for item in already_selected
    ]
    return (
        f"User question:\n{question}\n\n"
        f"Plan from turn 1:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        f"Already selected highlights from previous turns. Do not duplicate these:\n"
        f"{json.dumps(already, ensure_ascii=False)}\n\n"
        f"Selection pass: {pass_number}\n\n"
        f"Excerpts:\n{excerpt_text}"
    )


def legend_from_plan_and_candidates(plan: dict, candidates: list[dict]) -> list[dict]:
    legend = []
    seen = set()
    for category in plan.get("categories", []):
        label = (category.get("label") or "").strip()
        color = (category.get("color") or "").strip()
        if label and color and label not in seen:
            legend.append({"label": label, "color": color})
            seen.add(label)

    for item in candidates:
        label = (item.get("label") or "Relevant").strip()
        color = (item.get("color") or "#ffeb3b").strip()
        if label not in seen:
            legend.append({"label": label, "color": color})
            seen.add(label)

    return legend[:5]


async def run_agentic_highlights(question: str, model: str, api_key: str, extracted: list) -> dict:
    blocks = build_blocks(extracted)
    if not blocks:
        return {"mode": "highlights", "legend": [], "highlights": []}

    async with httpx.AsyncClient(timeout=120.0) as client:
        plan = await call_openrouter_json(
            client,
            model,
            api_key,
            AGENT_PLAN_PROMPT,
            f"User question:\n{question}\n\nPaper block index:\n{compact_paper_index(blocks)}",
            max_tokens=1200,
        )

        ranked_blocks = rank_blocks(blocks, plan)[:40]
        batches = [ranked_blocks[i:i + 8] for i in range(0, len(ranked_blocks), 8)]
        selected: list[dict] = []

        # Minimum 3 model turns: plan + two duplicate-aware selection passes.
        # Maximum 5 model turns: plan + up to four selection passes. Final JSON is
        # assembled by code from deduped candidates so long outputs cannot break JSON.
        required_batches = [batches[0] if len(batches) > 0 else [], batches[1] if len(batches) > 1 else []]
        for pass_number, batch in enumerate(required_batches, start=1):
            result = await call_openrouter_json(
                client,
                model,
                api_key,
                AGENT_CANDIDATE_PROMPT,
                make_candidate_user_message(question, plan, batch, selected, pass_number),
                max_tokens=2600,
            )
            selected = dedupe_candidates(selected + result.get("candidates", []), blocks)

        pass_number = 3
        while len(selected) < 12 and pass_number <= 4 and len(batches) >= pass_number:
            result = await call_openrouter_json(
                client,
                model,
                api_key,
                AGENT_CANDIDATE_PROMPT,
                make_candidate_user_message(question, plan, batches[pass_number - 1], selected, pass_number),
                max_tokens=2600,
            )
            selected = dedupe_candidates(selected + result.get("candidates", []), blocks)
            pass_number += 1

    enriched_highlights = []
    seen_blocks = set()
    seen_texts = set()
    for hl in selected:
        match_text = (hl.get("text") or "").strip()
        page = hl.get("page", 1)
        color = hl.get("color", "#ffeb3b")
        block_info = find_text_in_blocks(blocks, match_text, page)
        norm = normalize_text(match_text)
        block_key = block_info["id"] if block_info else f"page-{page}:{norm[:80]}"
        if not norm or norm in seen_texts or block_key in seen_blocks:
            continue
        seen_texts.add(norm)
        seen_blocks.add(block_key)
        enriched_highlights.append({
            "text": match_text,
            "page": block_info["page"] if block_info else page,
            "bbox": block_info["bbox"] if block_info else None,
            "color": color,
        })

    return {
        "mode": "highlights",
        "legend": legend_from_plan_and_candidates(plan, selected),
        "highlights": enriched_highlights,
    }
