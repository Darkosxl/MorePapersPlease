import json
import uuid
from pathlib import Path

import fitz
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="MorePapersPlease")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
TEXT_DIR = DATA_DIR / "text"
PDF_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HIGHLIGHTS_PROMPT = """You are an academic research assistant embedded in a PDF reader. Find passages in the paper that answer the user's question and color-code them by category. Do NOT write any commentary or summary — the user wants to read the highlights themselves without bias.

You receive the full text of a paper, organized by page. The user asks a question.

Respond ONLY with valid JSON. No markdown, no code fences, no extra text. Schema:

{
  "legend": [
    {"label": "Short label", "color": "#hexcolor"}
  ],
  "highlights": [
    {
      "text": "Exact verbatim substring from the paper",
      "page": 1,
      "color": "#hexcolor (must match one from legend)"
    }
  ]
}

Rules:
- 2-5 distinct colors max, each with a meaningful category label.
- "text" must be an EXACT verbatim substring from the paper. Copy precisely, do not paraphrase.
- Each highlight needs the correct page number.
- Include as many highlights as the question genuinely requires; err on the side of comprehensive.
- If nothing relevant exists, return empty arrays.
- NO answer field, NO commentary, NO annotations. Highlights only."""

COMMENTARY_PROMPT = """You are an academic research assistant. The user has selected a passage from a paper and is asking a question about that passage specifically. Provide a focused commentary based on the selected passage.

Respond ONLY with valid JSON. No markdown, no code fences, no extra text. Schema:

{
  "answer": "Your commentary, 2-6 sentences, grounded strictly in the selected passage."
}"""


class AskRequest(BaseModel):
    pdf_id: str
    question: str
    model: str
    api_key: str
    mode: str = "highlights"
    selected_text: str | None = None


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    pdf_id = uuid.uuid4().hex[:12]
    pdf_path = PDF_DIR / f"{pdf_id}.pdf"
    text_path = TEXT_DIR / f"{pdf_id}.json"

    content = await file.read()
    pdf_path.write_bytes(content)

    extracted = []
    doc = fitz.open(stream=content, filetype="pdf")
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        page_blocks = []
        for block in blocks:
            x0, y0, x1, y1, text, block_type, block_no = block
            text = text.strip()
            if text and block_type == 0:
                page_blocks.append({
                    "text": text,
                    "bbox": [x0, y0, x1, y1],
                    "page": page_num,
                })
        extracted.append({"page": page_num, "blocks": page_blocks})
    doc.close()

    text_path.write_text(json.dumps(extracted, ensure_ascii=False))

    full_text = ""
    total_blocks = 0
    for page_data in extracted:
        full_text += f"\n--- PAGE {page_data['page']} ---\n"
        for block in page_data["blocks"]:
            full_text += block["text"] + "\n"
            total_blocks += 1

    return {
        "pdf_id": pdf_id,
        "filename": file.filename,
        "page_count": len(extracted),
        "block_count": total_blocks,
        "char_count": len(full_text),
        "text_preview": full_text[:500],
    }


@app.get("/api/pdf/{pdf_id}")
async def get_pdf(pdf_id: str):
    pdf_path = PDF_DIR / f"{pdf_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf")


@app.get("/api/text/{pdf_id}")
async def get_text(pdf_id: str):
    text_path = TEXT_DIR / f"{pdf_id}.json"
    if not text_path.exists():
        raise HTTPException(404, "Text data not found")
    return json.loads(text_path.read_text())


@app.delete("/api/pdf/{pdf_id}")
async def delete_pdf(pdf_id: str):
    pdf_path = PDF_DIR / f"{pdf_id}.pdf"
    text_path = TEXT_DIR / f"{pdf_id}.json"
    if pdf_path.exists():
        pdf_path.unlink()
    if text_path.exists():
        text_path.unlink()
    return {"status": "deleted"}


@app.post("/api/ask")
async def ask_question(req: AskRequest):
    text_path = TEXT_DIR / f"{req.pdf_id}.json"
    if not text_path.exists():
        raise HTTPException(404, "PDF text not found. Re-upload the PDF.")

    extracted = json.loads(text_path.read_text())

    if req.mode == "commentary":
        if not req.selected_text or not req.selected_text.strip():
            raise HTTPException(400, "Commentary mode requires selected_text.")
        system_prompt = COMMENTARY_PROMPT
        user_message = (
            f"Selected passage from the paper:\n\"\"\"\n{req.selected_text.strip()}\n\"\"\"\n\n"
            f"User question: {req.question}"
        )
    else:
        full_text = ""
        for page_data in extracted:
            full_text += f"\n--- PAGE {page_data['page']} ---\n"
            for block in page_data["blocks"]:
                full_text += block["text"] + "\n"
        system_prompt = HIGHLIGHTS_PROMPT
        user_message = f"Paper text:\n{full_text}\n\nUser question: {req.question}"

    headers = {
        "Authorization": f"Bearer {req.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "MorePapersPlease",
    }

    payload = {
        "model": req.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
    }

    result = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(3):
            response = await client.post(OPENROUTER_URL, json=payload, headers=headers)

            if response.status_code != 200:
                detail = response.text[:500]
                raise HTTPException(502, f"OpenRouter error ({response.status_code}): {detail}")

            data = response.json()

            # OpenRouter sometimes returns 200 with an embedded error
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
            content = (content or "").strip()
            if not content:
                finish = choices[0].get("finish_reason", "?")
                usage = data.get("usage", {})
                raise HTTPException(
                    502,
                    f"Model '{req.model}' returned empty content (finish_reason={finish}, usage={usage}). "
                    f"Possible causes: rate limit, no credits, content policy block, output truncation, "
                    f"or the model genuinely returned nothing. Check OpenRouter dashboard.",
                )

            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            try:
                result = json.loads(content)
                break
            except json.JSONDecodeError:
                if attempt == 2:
                    raise HTTPException(500, "AI returned invalid JSON")

    if result is None:
        raise HTTPException(500, "AI returned invalid JSON")

    if req.mode == "commentary":
        return {"mode": "commentary", "answer": result.get("answer", "")}

    enriched_highlights = []
    for hl in result.get("highlights", []):
        match_text = hl.get("text", "")
        page = hl.get("page", 1)
        color = hl.get("color", "#ffeb3b")

        block_info = find_text_in_extracted(extracted, match_text)
        enriched_highlights.append({
            "text": match_text,
            "page": block_info["page"] if block_info else page,
            "bbox": block_info["bbox"] if block_info else None,
            "color": color,
        })

    return {
        "mode": "highlights",
        "legend": result.get("legend", []),
        "highlights": enriched_highlights,
    }


def find_text_in_extracted(extracted: list, search_text: str) -> dict | None:
    search_clean = search_text.strip().lower()
    search_start = search_clean[:60]

    for page_data in extracted:
        for block in page_data.get("blocks", []):
            block_text = block["text"].strip().lower()
            if search_start in block_text:
                return {
                    "page": page_data["page"],
                    "bbox": block["bbox"],
                }

    for page_data in extracted:
        for block in page_data.get("blocks", []):
            block_text = block["text"].strip().lower()
            if any(word in block_text for word in search_clean.split()[:3] if len(word) > 4):
                return {
                    "page": page_data["page"],
                    "bbox": block["bbox"],
                }

    return None


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
