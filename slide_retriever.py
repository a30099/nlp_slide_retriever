from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import fitz
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())


def get_title(raw_text: str) -> str:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def extract_slides(pdf_dir: str) -> list[dict]:
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))
    slides: list[dict] = []

    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)

        for page_index in range(len(doc)):
            page = doc[page_index]
            raw_text = page.get_text("text")
            title = get_title(raw_text)
            page_no = page_index + 1

            slides.append(
                {
                    "slide_id": f"{pdf_path.stem}_p{page_no}",
                    "filename": pdf_path.name,
                    "page": page_no,
                    "title": title,
                    "raw_text": clean_text(raw_text),
                }
            )

        doc.close()

    return slides


def load_captions(index_dir: str) -> dict[str, str]:
    caption_path = Path(index_dir) / "vision_captions.json"
    if caption_path.exists():
        return json.loads(caption_path.read_text(encoding="utf-8"))
    return {}


def make_document_text(slide: dict, caption: str) -> str:
    return clean_text(
        f"""
        filename: {slide["filename"]}
        page: {slide["page"]}
        title: {slide["title"]}
        raw_text: {slide["raw_text"]}
        vision_caption: {caption}
        """
    )


def build_index(
    pdf_dir: str,
    output_dir: str,
    min_df: int,
    top_features: int,
) -> None:
    slides = extract_slides(pdf_dir)
    captions = load_captions(output_dir)

    for slide in slides:
        caption = captions.get(slide["slide_id"], "")
        slide["vision_caption"] = caption
        slide["document_text"] = make_document_text(slide, caption)

    texts = [slide["document_text"] for slide in slides]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        lowercase=True,
        min_df=min_df,
        max_features=top_features,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(texts)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "slides.json").write_text(
        json.dumps(slides, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    joblib.dump(vectorizer, output_path / "tfidf_vectorizer.joblib")
    joblib.dump(matrix, output_path / "tfidf_matrix.joblib")

    caption_count = sum(1 for slide in slides if slide["vision_caption"])

    print(f"Indexed slides: {len(slides)}")
    print(f"Slides with vision captions: {caption_count}")
    print(f"TF-IDF matrix shape: {matrix.shape}")
    print(f"Saved index to: {output_path}")


def render_slide_image(
    pdf_path: Path,
    page_index: int,
    image_path: Path,
    scale: float,
) -> None:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    pixmap.save(image_path)
    doc.close()


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def caption_one_slide(
    image_path: Path,
    raw_text: str,
    model: str,
) -> str:
    from openai import OpenAI

    client = OpenAI()
    base64_image = encode_image(image_path)

    prompt = f"""
You are creating retrieval notes for an NLP course slide.

Goal:
Make this slide easy to retrieve from English or Traditional Chinese questions.

Use the slide image and the extracted PDF text.

Extracted PDF text:
{raw_text[:4000]}

Return plain text only, using this exact structure:

visible_text:
<important visible words, labels, formulas, examples>

visual_summary:
<describe diagrams, screenshots, charts, images, and examples on the slide>

concept_keywords_en:
<English keywords, aliases, abbreviations>

concept_keywords_zh:
<Traditional Chinese keywords and likely student query terms>
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}",
                    },
                ],
            }
        ],
        max_output_tokens=260,
    )

    return clean_text(response.output_text)


def caption_slides(
    pdf_dir: str,
    index_dir: str,
    model: str,
    max_pages: int | None,
    scale: float,
) -> None:
    index_path = Path(index_dir)
    image_dir = index_path / "slide_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    slides = extract_slides(pdf_dir)
    captions = load_captions(index_dir)

    caption_path = index_path / "vision_captions.json"

    processed_candidates = 0

    for slide in slides:
        if max_pages is not None and processed_candidates >= max_pages:
            break

        processed_candidates += 1
        slide_id = slide["slide_id"]

        if slide_id in captions:
            print(f"Skip existing caption: {slide_id}")
            continue

        pdf_path = Path(pdf_dir) / slide["filename"]
        image_path = image_dir / f"{slide_id}.png"

        render_slide_image(
            pdf_path=pdf_path,
            page_index=slide["page"] - 1,
            image_path=image_path,
            scale=scale,
        )

        print(f"Captioning: {slide_id}")
        caption = caption_one_slide(
            image_path=image_path,
            raw_text=slide["raw_text"],
            model=model,
        )

        captions[slide_id] = caption

        caption_path.write_text(
            json.dumps(captions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Saved captions: {len(captions)}")
    print(f"Caption file: {caption_path}")

def normalize_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def build_embedding_index(
    index_dir: str,
    model: str,
    batch_size: int,
) -> None:
    from openai import OpenAI

    client = OpenAI()
    index_path = Path(index_dir)

    slides = json.loads((index_path / "slides.json").read_text(encoding="utf-8"))
    texts = [slide["document_text"] for slide in slides]

    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        response = client.embeddings.create(
            model=model,
            input=batch,
        )

        embeddings.extend([item.embedding for item in response.data])

        finished = min(start + batch_size, len(texts))
        print(f"Embedded {finished}/{len(texts)} slides")

    embedding_matrix = np.array(embeddings, dtype=np.float32)
    embedding_matrix = normalize_embedding_matrix(embedding_matrix)

    np.save(index_path / "embedding_matrix.npy", embedding_matrix)

    (index_path / "embedding_meta.json").write_text(
        json.dumps(
            {
                "model": model,
                "slides": len(slides),
                "dimension": int(embedding_matrix.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved embedding matrix: {embedding_matrix.shape}")
    print(f"Embedding model: {model}")


def embed_query(
    query: str,
    model: str,
) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()

    response = client.embeddings.create(
        model=model,
        input=[query],
    )

    query_vector = np.array(response.data[0].embedding, dtype=np.float32)
    query_vector = query_vector / max(np.linalg.norm(query_vector), 1e-12)

    return query_vector


def hybrid_search_index(
    index_dir: str,
    query: str,
    top_k: int,
    keyword_k: int,
    embedding_k: int,
    rrf_k: int,
) -> None:
    index_path = Path(index_dir)

    slides = json.loads((index_path / "slides.json").read_text(encoding="utf-8"))
    vectorizer = joblib.load(index_path / "tfidf_vectorizer.joblib")
    tfidf_matrix = joblib.load(index_path / "tfidf_matrix.joblib")

    embedding_matrix = np.load(index_path / "embedding_matrix.npy")
    embedding_meta = json.loads((index_path / "embedding_meta.json").read_text(encoding="utf-8"))
    embedding_model = embedding_meta["model"]

    query_tfidf = vectorizer.transform([query])
    keyword_scores = (tfidf_matrix @ query_tfidf.T).toarray().ravel()
    keyword_indices = np.argsort(keyword_scores)[::-1][:keyword_k]

    query_embedding = embed_query(query=query, model=embedding_model)
    embedding_scores = embedding_matrix @ query_embedding
    embedding_indices = np.argsort(embedding_scores)[::-1][:embedding_k]

    final_scores: dict[int, float] = {}
    keyword_ranks: dict[int, int] = {}
    embedding_ranks: dict[int, int] = {}

    for rank, idx in enumerate(keyword_indices, start=1):
        idx = int(idx)
        final_scores[idx] = final_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        keyword_ranks[idx] = rank

    for rank, idx in enumerate(embedding_indices, start=1):
        idx = int(idx)
        final_scores[idx] = final_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        embedding_ranks[idx] = rank

    final_indices = sorted(
        final_scores.keys(),
        key=lambda idx: final_scores[idx],
        reverse=True,
    )[:top_k]

    results = []

    for rank, idx in enumerate(final_indices, start=1):
        slide = slides[idx]

        evidence = clean_text(
            f"""
            raw_text: {slide["raw_text"]}
            vision_caption: {slide.get("vision_caption", "")}
            """
        )

        results.append(
            {
                "rank": rank,
                "rrf_score": round(float(final_scores[idx]), 6),
                "keyword_rank": keyword_ranks.get(idx),
                "keyword_score": round(float(keyword_scores[idx]), 6),
                "embedding_rank": embedding_ranks.get(idx),
                "embedding_score": round(float(embedding_scores[idx]), 6),
                "filename": slide["filename"],
                "page": slide["page"],
                "title": slide["title"],
                "evidence": evidence[:700] + ("..." if len(evidence) > 700 else ""),
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))

def get_hybrid_results(
    index_dir: str,
    query: str,
    top_k: int,
    keyword_k: int,
    embedding_k: int,
    rrf_k: int,
) -> list[dict]:
    index_path = Path(index_dir)

    slides = json.loads((index_path / "slides.json").read_text(encoding="utf-8"))
    vectorizer = joblib.load(index_path / "tfidf_vectorizer.joblib")
    tfidf_matrix = joblib.load(index_path / "tfidf_matrix.joblib")

    embedding_matrix = np.load(index_path / "embedding_matrix.npy")
    embedding_meta = json.loads((index_path / "embedding_meta.json").read_text(encoding="utf-8"))
    embedding_model = embedding_meta["model"]

    query_tfidf = vectorizer.transform([query])
    keyword_scores = (tfidf_matrix @ query_tfidf.T).toarray().ravel()
    keyword_indices = np.argsort(keyword_scores)[::-1][:keyword_k]

    query_embedding = embed_query(query=query, model=embedding_model)
    embedding_scores = embedding_matrix @ query_embedding
    embedding_indices = np.argsort(embedding_scores)[::-1][:embedding_k]

    final_scores: dict[int, float] = {}
    keyword_ranks: dict[int, int] = {}
    embedding_ranks: dict[int, int] = {}

    for rank, idx in enumerate(keyword_indices, start=1):
        idx = int(idx)
        final_scores[idx] = final_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        keyword_ranks[idx] = rank

    for rank, idx in enumerate(embedding_indices, start=1):
        idx = int(idx)
        final_scores[idx] = final_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        embedding_ranks[idx] = rank

    seed_indices = sorted(
        final_scores.keys(),
        key=lambda idx: final_scores[idx],
        reverse=True,
    )[:top_k]

    page_lookup = {
        (slide["filename"], slide["page"]): idx
        for idx, slide in enumerate(slides)
    }

    expanded_indices = []
    seen_indices = set()

    for idx in seed_indices:
        slide = slides[idx]

        candidate_indices = [
            idx,
            page_lookup.get((slide["filename"], slide["page"] + 1)),
            page_lookup.get((slide["filename"], slide["page"] - 1)),
        ]

        for candidate_idx in candidate_indices:
            if candidate_idx is not None and candidate_idx not in seen_indices:
                expanded_indices.append(candidate_idx)
                seen_indices.add(candidate_idx)

    results = []

    for rank, idx in enumerate(expanded_indices, start=1):
        slide = slides[idx]

        results.append(
            {
                "rank": rank,
                "rrf_score": round(float(final_scores.get(idx, 0.0)), 6),
                "keyword_rank": keyword_ranks.get(idx),
                "keyword_score": round(float(keyword_scores[idx]), 6),
                "embedding_rank": embedding_ranks.get(idx),
                "embedding_score": round(float(embedding_scores[idx]), 6),
                "slide_id": slide["slide_id"],
                "filename": slide["filename"],
                "page": slide["page"],
                "title": slide["title"],
                "raw_text": slide["raw_text"],
                "vision_caption": slide.get("vision_caption", ""),
            }
        )

    return results

def make_answer_context(results: list[dict]) -> str:
    chunks = []

    for item in results:
        chunk = f"""
[Slide {item["rank"]}]
filename: {item["filename"]}
page: {item["page"]}
title: {item["title"]}
retrieval_info: keyword_rank={item["keyword_rank"]}, embedding_rank={item["embedding_rank"]}
raw_text:
{item["raw_text"]}

vision_caption:
{item["vision_caption"]}
"""
        chunks.append(clean_text(chunk))

    return "\n\n".join(chunks)


def answer_question(
    index_dir: str,
    query: str,
    top_k: int,
    keyword_k: int,
    embedding_k: int,
    rrf_k: int,
    model: str,
) -> None:
    from openai import OpenAI

    results = get_hybrid_results(
        index_dir=index_dir,
        query=query,
        top_k=top_k,
        keyword_k=keyword_k,
        embedding_k=embedding_k,
        rrf_k=rrf_k,
    )

    context = make_answer_context(results)

    prompt = f"""
你是一個 NLP 課程投影片檢索問答系統。

請嚴格遵守：
1. 只能根據下方 Retrieved Slides 的文字與隨後提供的投影片圖片回答。
2. 不要使用外部知識補答案。
3. 如果 Retrieved Slides 的文字和圖片都沒有足夠證據，請回答「找不到足夠證據」。
4. 答案要簡潔。
5. 每個答案都必須附上來源 filename 與 page。
6. evidence 要引用投影片中能支持答案的重點，不要編造。
7. 如果問題在問圖片中的動漫、卡通、遊戲等虛構角色，可以根據投影片圖片辨識角色名稱。
8. 如果使用者把 NLP / LDA 等相近詞打錯，請根據檢索到的最相關投影片判斷，但答案仍然只能根據投影片文字與圖片。

User Question:
{query}

Retrieved Slides:
{context}

接下來會依照 Slide 1, Slide 2, ... 的順序提供投影片圖片。

請用以下格式輸出：

答案：
<答案>

來源：
1. <filename>, p.<page> — <evidence>
"""

    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    image_dir = Path(index_dir) / "slide_images"

    for item in results:
        image_path = image_dir / f'{item["slide_id"]}.png'

        content.append(
            {
                "type": "input_text",
                "text": f'Image for Slide {item["rank"]}: {item["filename"]}, p.{item["page"]}, title: {item["title"]}',
            }
        )

        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{encode_image(image_path)}",
            }
        )

    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_output_tokens=700,
    )

    print(response.output_text)

    print("\n--- Retrieved slides used ---")
    compact_results = []

    for item in results:
        compact_results.append(
            {
                "rank": item["rank"],
                "rrf_score": item["rrf_score"],
                "keyword_rank": item["keyword_rank"],
                "embedding_rank": item["embedding_rank"],
                "filename": item["filename"],
                "page": item["page"],
                "title": item["title"],
            }
        )

    print(json.dumps(compact_results, ensure_ascii=False, indent=2))

def search_index(
    index_dir: str,
    query: str,
    top_k: int,
) -> None:
    index_path = Path(index_dir)

    slides = json.loads((index_path / "slides.json").read_text(encoding="utf-8"))
    vectorizer = joblib.load(index_path / "tfidf_vectorizer.joblib")
    matrix = joblib.load(index_path / "tfidf_matrix.joblib")

    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        slide = slides[int(idx)]
        evidence = clean_text(
            f"""
            raw_text: {slide["raw_text"]}
            vision_caption: {slide.get("vision_caption", "")}
            """
        )

        results.append(
            {
                "rank": rank,
                "score": round(float(scores[idx]), 6),
                "filename": slide["filename"],
                "page": slide["page"],
                "title": slide["title"],
                "evidence": evidence[:700] + ("..." if len(evidence) > 700 else ""),
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--pdf_dir", required=True)
    build_parser.add_argument("--output_dir", required=True)
    build_parser.add_argument("--min_df", type=int, default=1)
    build_parser.add_argument("--top_features", type=int, default=100000)

    caption_parser = subparsers.add_parser("caption")
    caption_parser.add_argument("--pdf_dir", required=True)
    caption_parser.add_argument("--index_dir", required=True)
    caption_parser.add_argument("--model", default="gpt-4.1-mini")
    caption_parser.add_argument("--max_pages", type=int, default=None)
    caption_parser.add_argument("--scale", type=float, default=1.5)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--index_dir", required=True)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top_k", type=int, default=5)

    embed_parser = subparsers.add_parser("embed")
    embed_parser.add_argument("--index_dir", required=True)
    embed_parser.add_argument("--model", default="text-embedding-3-large")
    embed_parser.add_argument("--batch_size", type=int, default=64)

    hybrid_parser = subparsers.add_parser("hybrid_search")
    hybrid_parser.add_argument("--index_dir", required=True)
    hybrid_parser.add_argument("--query", required=True)
    hybrid_parser.add_argument("--top_k", type=int, default=5)
    hybrid_parser.add_argument("--keyword_k", type=int, default=30)
    hybrid_parser.add_argument("--embedding_k", type=int, default=30)
    hybrid_parser.add_argument("--rrf_k", type=int, default=60)

    answer_parser = subparsers.add_parser("answer")
    answer_parser.add_argument("--index_dir", required=True)
    answer_parser.add_argument("--query", required=True)
    answer_parser.add_argument("--top_k", type=int, default=5)
    answer_parser.add_argument("--keyword_k", type=int, default=30)
    answer_parser.add_argument("--embedding_k", type=int, default=30)
    answer_parser.add_argument("--rrf_k", type=int, default=60)
    answer_parser.add_argument("--model", default="gpt-4.1")

    args = parser.parse_args()

    if args.command == "build":
        build_index(
            pdf_dir=args.pdf_dir,
            output_dir=args.output_dir,
            min_df=args.min_df,
            top_features=args.top_features,
        )

    if args.command == "caption":
        caption_slides(
            pdf_dir=args.pdf_dir,
            index_dir=args.index_dir,
            model=args.model,
            max_pages=args.max_pages,
            scale=args.scale,
        )

    if args.command == "search":
        search_index(
            index_dir=args.index_dir,
            query=args.query,
            top_k=args.top_k,
        )
    if args.command == "embed":
        build_embedding_index(
            index_dir=args.index_dir,
            model=args.model,
            batch_size=args.batch_size,
        )

    if args.command == "hybrid_search":
        hybrid_search_index(
            index_dir=args.index_dir,
            query=args.query,
            top_k=args.top_k,
            keyword_k=args.keyword_k,
            embedding_k=args.embedding_k,
            rrf_k=args.rrf_k,
        )
    if args.command == "answer":
        answer_question(
            index_dir=args.index_dir,
            query=args.query,
            top_k=args.top_k,
            keyword_k=args.keyword_k,
            embedding_k=args.embedding_k,
            rrf_k=args.rrf_k,
            model=args.model,
        )


if __name__ == "__main__":
    main()