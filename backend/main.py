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

from highlight_agent import call_openrouter_json, run_agentic_highlights


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
        user_message = (
            f"Selected passage from the paper:\n\"\"\"\n{req.selected_text.strip()}\n\"\"\"\n\n"
            f"User question: {req.question}"
        )
        async with httpx.AsyncClient(timeout=120.0) as client:
            result = await call_openrouter_json(
                client,
                req.model,
                req.api_key,
                COMMENTARY_PROMPT,
                user_message,
                max_tokens=1200,
                temperature=0.3,
            )
        return {"mode": "commentary", "answer": result.get("answer", "")}

    return await run_agentic_highlights(req.question, req.model, req.api_key, extracted)


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
