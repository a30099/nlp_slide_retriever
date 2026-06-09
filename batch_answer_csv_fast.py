from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np


VISUAL_KEYWORDS = [
    "圖片", "圖中", "圖裡", "照片", "截圖", "流程圖", "diagram", "figure", "image", "chart", "表格", "table",
    "動漫", "人物", "角色", "卡通", "小智", "Pokemon", "Pokémon", "誰",
]


def clean_text(text: str) -> str:
    return " ".join(str(text).replace("\x00", " ").split())


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def read_csv_rows(input_csv: Path, has_header: bool) -> tuple[list[list[str]], list[str] | None, str]:
    encodings = ["utf-8-sig", "utf-8", "cp950", "big5"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with input_csv.open("r", encoding=encoding, newline="") as f:
                rows = list(csv.reader(f))
            header = rows[0] if has_header and rows else None
            body = rows[1:] if has_header and rows else rows
            return body, header, encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"Cannot read CSV with common encodings: {last_error}")


def write_csv_rows(output_csv: Path, rows: list[list[str]], header: list[str] | None) -> None:
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if header is not None:
            if len(header) == 1:
                writer.writerow([header[0], "answer"])
            else:
                writer.writerow(header)
        writer.writerows(rows)


def load_existing_answers(output_csv: Path, has_header: bool) -> dict[str, str]:
    if not output_csv.exists():
        return {}

    rows, _header, _encoding = read_csv_rows(output_csv, has_header=has_header)
    answers: dict[str, str] = {}
    for row in rows:
        if len(row) >= 2:
            q = row[0].strip()
            a = row[1].strip()
            if q and a and not a.startswith("ERROR:"):
                answers[q] = a
    return answers


def embed_queries(queries: list[str], model: str, batch_size: int) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    all_vectors: list[list[float]] = []

    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_vectors.extend([item.embedding for item in response.data])
        print(f"Embedded queries {min(start + batch_size, len(queries))}/{len(queries)}")

    return normalize_rows(np.asarray(all_vectors, dtype=np.float32))


def rrf_indices(
    keyword_indices: np.ndarray,
    embedding_indices: np.ndarray,
    keyword_k: int,
    embedding_k: int,
    rrf_k: int,
) -> tuple[list[int], dict[int, float], dict[int, int], dict[int, int]]:
    final_scores: dict[int, float] = {}
    keyword_ranks: dict[int, int] = {}
    embedding_ranks: dict[int, int] = {}

    for rank, idx in enumerate(keyword_indices[:keyword_k], start=1):
        idx = int(idx)
        final_scores[idx] = final_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        keyword_ranks[idx] = rank

    for rank, idx in enumerate(embedding_indices[:embedding_k], start=1):
        idx = int(idx)
        final_scores[idx] = final_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        embedding_ranks[idx] = rank

    merged = sorted(final_scores.keys(), key=lambda idx: final_scores[idx], reverse=True)
    return merged, final_scores, keyword_ranks, embedding_ranks


def expand_neighbors(slides: list[dict[str, Any]], seed_indices: list[int], neighbor: int) -> list[int]:
    page_lookup = {(slide["filename"], int(slide["page"])): idx for idx, slide in enumerate(slides)}
    expanded: list[int] = []
    seen: set[int] = set()

    for idx in seed_indices:
        slide = slides[idx]
        filename = slide["filename"]
        page = int(slide["page"])

        candidates: list[int | None] = [idx]
        for offset in range(1, neighbor + 1):
            candidates.append(page_lookup.get((filename, page + offset)))
            candidates.append(page_lookup.get((filename, page - offset)))

        for c in candidates:
            if c is not None and c not in seen:
                expanded.append(c)
                seen.add(c)

    return expanded


def make_context(slides: list[dict[str, Any]], indices: list[int]) -> str:
    chunks = []
    for rank, idx in enumerate(indices, start=1):
        slide = slides[idx]
        chunk = f"""
[Slide {rank}]
filename: {slide['filename']}
page: {slide['page']}
title: {slide.get('title', '')}
raw_text:
{slide.get('raw_text', '')}

vision_caption:
{slide.get('vision_caption', '')}
"""
        chunks.append(clean_text(chunk))
    return "\n\n".join(chunks)


def should_use_image(query: str, image_mode: str) -> bool:
    if image_mode == "always":
        return True
    if image_mode == "never":
        return False
    q = query.lower()
    return any(keyword.lower() in q for keyword in VISUAL_KEYWORDS)


def call_answer_model(
    query: str,
    context: str,
    slides: list[dict[str, Any]],
    indices_for_images: list[int],
    index_dir: Path,
    model: str,
    image_mode: str,
    image_limit: int,
    max_output_tokens: int,
    retry_count: int,
    retry_base_sleep: float,
) -> str:
    from openai import OpenAI

    client = OpenAI()

    prompt = f"""
你是一個 NLP 課程投影片問答系統。請只根據 Retrieved Slides 回答。

規則：
1. 只輸出「答案本身」，不要輸出來源、解釋、前綴。
2. 如果答案是日期、數字、名詞，請盡量短。
3. 如果 Retrieved Slides 沒有足夠證據，回答：找不到足夠證據。
4. 可以根據 vision_caption 裡的圖片描述回答；只有在另外提供圖片時，才可直接看圖片。
5. 不要使用外部知識補答案。

User Question:
{query}

Retrieved Slides:
{context}

只輸出答案本身：
"""

    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]

    if image_limit > 0 and should_use_image(query, image_mode):
        image_dir = index_dir / "slide_images"
        count = 0
        for idx in indices_for_images:
            if count >= image_limit:
                break
            slide = slides[idx]
            image_path = image_dir / f"{slide['slide_id']}.png"
            if not image_path.exists():
                continue
            content.append({
                "type": "input_text",
                "text": f"Image for {slide['filename']}, p.{slide['page']}, title: {slide.get('title', '')}",
            })
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{encode_image(image_path)}",
            })
            count += 1

    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                max_output_tokens=max_output_tokens,
            )
            return clean_text(response.output_text)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if "429" not in message and "RateLimit" not in message and attempt >= retry_count:
                break
            sleep_seconds = retry_base_sleep * (2 ** attempt)
            print(f"Rate/error; retry after {sleep_seconds:.1f}s: {type(exc).__name__}: {exc}")
            time.sleep(sleep_seconds)

    return f"ERROR: {type(last_error).__name__}: {last_error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--index_dir", default="index")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--has_header", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--neighbor", type=int, default=1)
    parser.add_argument("--keyword_k", type=int, default=25)
    parser.add_argument("--embedding_k", type=int, default=25)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--query_embed_batch_size", type=int, default=64)
    parser.add_argument("--image_mode", choices=["never", "auto", "always"], default="auto")
    parser.add_argument("--image_limit", type=int, default=1)
    parser.add_argument("--max_output_tokens", type=int, default=100)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    parser.add_argument("--retry_count", type=int, default=4)
    parser.add_argument("--retry_base_sleep", type=float, default=4.0)
    parser.add_argument("--debug_jsonl", default="batch_fast_debug.jsonl")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    index_dir = Path(args.index_dir)

    rows, header, encoding = read_csv_rows(input_csv, has_header=args.has_header)
    print(f"Read {len(rows)} rows from {input_csv} using encoding={encoding}")

    existing_answers = load_existing_answers(output_csv, has_header=args.has_header) if args.resume else {}

    questions = [row[0].strip() if row else "" for row in rows]
    questions_to_embed = [q if q else " " for q in questions]

    slides = json.loads((index_dir / "slides.json").read_text(encoding="utf-8"))
    vectorizer = joblib.load(index_dir / "tfidf_vectorizer.joblib")
    tfidf_matrix = joblib.load(index_dir / "tfidf_matrix.joblib")
    embedding_matrix = np.load(index_dir / "embedding_matrix.npy")
    embedding_meta = json.loads((index_dir / "embedding_meta.json").read_text(encoding="utf-8"))
    embedding_model = embedding_meta["model"]

    query_tfidf = vectorizer.transform(questions_to_embed)
    keyword_scores_all = (query_tfidf @ tfidf_matrix.T).toarray()

    query_embeddings = embed_queries(
        queries=questions_to_embed,
        model=embedding_model,
        batch_size=args.query_embed_batch_size,
    )
    embedding_scores_all = query_embeddings @ embedding_matrix.T

    output_rows: list[list[str]] = []
    debug_path = Path(args.debug_jsonl)
    debug_file = debug_path.open("a", encoding="utf-8")

    try:
        for i, row in enumerate(rows):
            question = questions[i]
            if not question:
                output_rows.append(["", ""])
                continue

            if args.resume and question in existing_answers:
                answer = existing_answers[question]
                print(f"[{i+1}/{len(rows)}] SKIP existing: {question} -> {answer}")
                new_row = row[:]
                if len(new_row) < 2:
                    new_row += [""] * (2 - len(new_row))
                new_row[1] = answer
                output_rows.append(new_row)
                continue

            keyword_scores = keyword_scores_all[i]
            embedding_scores = embedding_scores_all[i]
            keyword_indices = np.argsort(keyword_scores)[::-1]
            embedding_indices = np.argsort(embedding_scores)[::-1]

            merged, final_scores, keyword_ranks, embedding_ranks = rrf_indices(
                keyword_indices=keyword_indices,
                embedding_indices=embedding_indices,
                keyword_k=args.keyword_k,
                embedding_k=args.embedding_k,
                rrf_k=args.rrf_k,
            )

            seed_indices = merged[: args.top_k]
            context_indices = expand_neighbors(slides, seed_indices, neighbor=args.neighbor)
            context = make_context(slides, context_indices)

            print(f"[{i+1}/{len(rows)}] {question}")
            answer = call_answer_model(
                query=question,
                context=context,
                slides=slides,
                indices_for_images=seed_indices,
                index_dir=index_dir,
                model=args.model,
                image_mode=args.image_mode,
                image_limit=args.image_limit,
                max_output_tokens=args.max_output_tokens,
                retry_count=args.retry_count,
                retry_base_sleep=args.retry_base_sleep,
            )
            print(f" -> {answer}")

            new_row = row[:]
            if len(new_row) < 2:
                new_row += [""] * (2 - len(new_row))
            new_row[1] = answer
            output_rows.append(new_row)

            debug_record = {
                "question": question,
                "answer": answer,
                "used_images": should_use_image(question, args.image_mode) and args.image_limit > 0,
                "seed_slides": [
                    {
                        "filename": slides[idx]["filename"],
                        "page": slides[idx]["page"],
                        "title": slides[idx].get("title", ""),
                        "rrf_score": round(float(final_scores.get(idx, 0.0)), 6),
                        "keyword_rank": keyword_ranks.get(idx),
                        "embedding_rank": embedding_ranks.get(idx),
                    }
                    for idx in seed_indices
                ],
                "context_slides": [
                    {"filename": slides[idx]["filename"], "page": slides[idx]["page"], "title": slides[idx].get("title", "")}
                    for idx in context_indices
                ],
            }
            debug_file.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
            debug_file.flush()

            write_csv_rows(output_csv, output_rows, header)
            time.sleep(args.sleep_seconds)
    finally:
        debug_file.close()

    write_csv_rows(output_csv, output_rows, header)
    print(f"Saved output CSV: {output_csv}")
    print(f"Saved debug file: {debug_path}")


if __name__ == "__main__":
    main()
