from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from openai import OpenAI

from slide_retriever import (
    clean_text,
    encode_image,
    get_hybrid_results,
    make_answer_context,
)


def read_csv_rows(path: Path) -> tuple[list[list[str]], str]:
    """Read CSV with common encodings used by Excel / Windows Traditional Chinese."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.reader(f)), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Cannot decode CSV: {path}. Last error: {last_error}")


def write_csv_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def expand_neighbor_results(
    index_dir: str,
    results: list[dict],
    neighbor_pages: int,
    max_context_slides: int,
) -> list[dict]:
    """Add previous/next pages from the same PDF to reduce missed adjacent evidence."""
    if neighbor_pages <= 0:
        return results[:max_context_slides]

    index_path = Path(index_dir)
    slides = json.loads((index_path / "slides.json").read_text(encoding="utf-8"))

    by_file_page = {
        (slide["filename"], int(slide["page"])): slide
        for slide in slides
    }

    expanded: list[dict] = []
    seen: set[str] = set()

    def add_slide(slide: dict, source_item: dict | None = None) -> None:
        slide_id = slide["slide_id"]
        if slide_id in seen:
            return
        seen.add(slide_id)

        if source_item is None:
            source_item = {}

        expanded.append(
            {
                "rank": len(expanded) + 1,
                "rrf_score": source_item.get("rrf_score", 0.0),
                "keyword_rank": source_item.get("keyword_rank"),
                "keyword_score": source_item.get("keyword_score", 0.0),
                "embedding_rank": source_item.get("embedding_rank"),
                "embedding_score": source_item.get("embedding_score", 0.0),
                "slide_id": slide["slide_id"],
                "filename": slide["filename"],
                "page": slide["page"],
                "title": slide["title"],
                "raw_text": slide["raw_text"],
                "vision_caption": slide.get("vision_caption", ""),
            }
        )

    for item in results:
        # Add the retrieved page first.
        base_slide = by_file_page.get((item["filename"], int(item["page"])))
        if base_slide is not None:
            add_slide(base_slide, item)

        # Add next page before previous page because many lecture decks put the answer after the setup page.
        for offset in range(1, neighbor_pages + 1):
            next_slide = by_file_page.get((item["filename"], int(item["page"]) + offset))
            if next_slide is not None:
                add_slide(next_slide, item)

        for offset in range(1, neighbor_pages + 1):
            prev_slide = by_file_page.get((item["filename"], int(item["page"]) - offset))
            if prev_slide is not None:
                add_slide(prev_slide, item)

        if len(expanded) >= max_context_slides:
            break

    return expanded[:max_context_slides]


def answer_one_question(
    client: OpenAI,
    index_dir: str,
    question: str,
    model: str,
    top_k: int,
    keyword_k: int,
    embedding_k: int,
    rrf_k: int,
    neighbor_pages: int,
    max_context_slides: int,
    image_limit: int,
    max_output_tokens: int,
) -> tuple[str, list[dict], str]:
    results = get_hybrid_results(
        index_dir=index_dir,
        query=question,
        top_k=top_k,
        keyword_k=keyword_k,
        embedding_k=embedding_k,
        rrf_k=rrf_k,
    )
    results = expand_neighbor_results(
        index_dir=index_dir,
        results=results,
        neighbor_pages=neighbor_pages,
        max_context_slides=max_context_slides,
    )

    context = make_answer_context(results)

    prompt = f"""
你是一個 NLP 課程投影片檢索問答系統。現在要產生 CSV 的第 2 欄答案。

請嚴格遵守：
1. 只能根據 Retrieved Slides 的文字、vision_caption、以及後面提供的投影片圖片回答。
2. 不要輸出來源、頁碼、解釋、前綴或「答案：」。
3. 只輸出最終答案本身，越短越好。
4. 中文題目用中文回答；英文專有名詞可以保留英文。
5. 如果答案是數字、日期、公式、專有名詞，請直接輸出該答案。
6. 如果題目問「為什麼」，可以用一到兩句簡短回答。
7. 如果題目問圖片中的動漫、卡通、遊戲等角色，可以根據投影片圖片辨識角色名稱。
8. 如果文字和圖片都沒有足夠證據，才輸出：找不到足夠證據

Question:
{question}

Retrieved Slides:
{context}
"""

    content: list[dict] = [{"type": "input_text", "text": prompt}]
    image_dir = Path(index_dir) / "slide_images"

    for item in results[:image_limit]:
        image_path = image_dir / f'{item["slide_id"]}.png'
        content.append(
            {
                "type": "input_text",
                "text": f'Image for Slide {item["rank"]}: {item["filename"]}, p.{item["page"]}, title: {item["title"]}',
            }
        )
        if image_path.exists():
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encode_image(image_path)}",
                }
            )
        else:
            content.append(
                {
                    "type": "input_text",
                    "text": f"[Warning: image file missing: {image_path}]",
                }
            )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=max_output_tokens,
    )

    raw_answer = response.output_text.strip()
    answer = clean_text(raw_answer)

    # Avoid common prefixes in case the model still adds them.
    for prefix in ("答案：", "答案:", "Answer:", "answer:"):
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    return answer, results, raw_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--index_dir", default="index")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--keyword_k", type=int, default=30)
    parser.add_argument("--embedding_k", type=int, default=30)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--neighbor_pages", type=int, default=1)
    parser.add_argument("--max_context_slides", type=int, default=12)
    parser.add_argument("--image_limit", type=int, default=8)
    parser.add_argument("--max_output_tokens", type=int, default=160)
    parser.add_argument("--has_header", action="store_true")
    parser.add_argument("--debug_jsonl", default="batch_debug.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    rows, encoding = read_csv_rows(input_path)
    print(f"Read {len(rows)} rows from {input_path} using encoding={encoding}")

    client = OpenAI()
    output_rows: list[list[str]] = []
    debug_path = Path(args.debug_jsonl)

    with debug_path.open("w", encoding="utf-8") as debug_file:
        start_idx = 0
        if args.has_header and rows:
            header = list(rows[0])
            if len(header) < 2:
                header.append("answer")
            else:
                header[1] = header[1] or "answer"
            output_rows.append(header)
            start_idx = 1

        total_questions = sum(1 for row in rows[start_idx:] if row and row[0].strip())
        done = 0

        for row_idx, row in enumerate(rows[start_idx:], start=start_idx + 1):
            if not row or not row[0].strip():
                output_rows.append(row)
                continue

            question = row[0].strip()
            done += 1
            print(f"[{done}/{total_questions}] {question}")

            try:
                answer, results, raw_answer = answer_one_question(
                    client=client,
                    index_dir=args.index_dir,
                    question=question,
                    model=args.model,
                    top_k=args.top_k,
                    keyword_k=args.keyword_k,
                    embedding_k=args.embedding_k,
                    rrf_k=args.rrf_k,
                    neighbor_pages=args.neighbor_pages,
                    max_context_slides=args.max_context_slides,
                    image_limit=args.image_limit,
                    max_output_tokens=args.max_output_tokens,
                )
            except Exception as exc:
                answer = f"ERROR: {type(exc).__name__}: {exc}"
                results = []
                raw_answer = answer

            out_row = list(row)
            while len(out_row) < 2:
                out_row.append("")
            out_row[1] = answer
            output_rows.append(out_row)

            debug_record = {
                "row": row_idx,
                "question": question,
                "answer": answer,
                "raw_answer": raw_answer,
                "retrieved": [
                    {
                        "rank": item.get("rank"),
                        "filename": item.get("filename"),
                        "page": item.get("page"),
                        "title": item.get("title"),
                        "keyword_rank": item.get("keyword_rank"),
                        "embedding_rank": item.get("embedding_rank"),
                    }
                    for item in results
                ],
            }
            debug_file.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
            debug_file.flush()

            print(f" -> {answer}")
            write_csv_rows(output_path, output_rows)

    write_csv_rows(output_path, output_rows)
    print(f"Saved output CSV: {output_path}")
    print(f"Saved debug file: {debug_path}")


if __name__ == "__main__":
    main()
