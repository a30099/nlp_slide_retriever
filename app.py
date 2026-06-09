from __future__ import annotations

from pathlib import Path
from typing import Iterator

import streamlit as st
from openai import OpenAI

from slide_retriever import (
    encode_image,
    get_hybrid_results,
    make_answer_context,
)


INDEX_DIR = "index"
DEFAULT_MODEL = "gpt-4.1-mini"
TOP_K = 5
KEYWORD_K = 30
EMBEDDING_K = 30
RRF_K = 60


def stream_answer(
    content: list[dict],
    model: str,
) -> Iterator[str]:
    client = OpenAI()

    with client.responses.stream(
        model=model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_output_tokens=700,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta


def answer_question_for_ui(
    query: str,
    index_dir: str,
    model: str,
    top_k: int,
    keyword_k: int,
    embedding_k: int,
    rrf_k: int,
) -> tuple[Iterator[str], list[dict]]:
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
9. 請務必使用與「User Question」完全相同的語言來撰寫你的<答案>與<evidence>。

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

    return stream_answer(content, model), results


def show_retrieved_slide(item: dict, index_dir: str) -> None:
    image_path = Path(index_dir) / "slide_images" / f'{item["slide_id"]}.png'

    st.markdown(
        f"""
**Rank {item["rank"]}｜{item["filename"]}, p.{item["page"]}**  
Title: `{item["title"]}`  
RRF score: `{item["rrf_score"]}` ｜ Keyword rank: `{item["keyword_rank"]}` ｜ Embedding rank: `{item["embedding_rank"]}`
"""
    )

    st.image(str(image_path), caption=f'{item["filename"]}, p.{item["page"]}')

    with st.expander("查看 raw_text / vision_caption"):
        st.markdown("**raw_text**")
        st.text(item["raw_text"])

        st.markdown("**vision_caption**")
        st.text(item["vision_caption"])


def main() -> None:
    st.set_page_config(
        page_title="NLP Slide Retriever",
        page_icon="🔎",
        layout="wide",
    )

    st.title("🔎 NLP 課程投影片檢索問答系統")
    st.caption("Hybrid Retrieval = TF-IDF + OpenAI Embedding + Vision Caption + LLM Answer")
    st.info("請從範例問題選擇一題，或自行輸入問題。系統會回傳答案、來源 PDF、頁碼與檢索到的投影片。")

    with st.sidebar:
        st.header("系統設定")
        model = st.selectbox(
            "回答模型",
            ["gpt-4.1-mini", "gpt-4.1"],
            index=0,
        )

        top_k = st.slider("Top-K retrieved slides", 3, 10, TOP_K)
        keyword_k = st.slider("Keyword retrieval K", 10, 50, KEYWORD_K)
        embedding_k = st.slider("Embedding retrieval K", 10, 50, EMBEDDING_K)

        st.markdown("---")
        st.markdown("**目前索引資料夾**")
        st.code(INDEX_DIR)

    example_questions = [
        "什麼是 Time-homogeneous Markov process？",
        "Substitution Cipher 的總可能組合數大約是多少？",
        "幾月幾號是期中考？",
    ]

    selected_example = st.selectbox(
        "範例問題",
        [""] + example_questions,
    )

    default_question = selected_example if selected_example else ""

    with st.form("question_form"):
        query = st.text_area(
            "請輸入問題",
            value=default_question,
            height=100,
            placeholder="例如：什麼是 Time-homogeneous Markov process？",
        )

        submitted = st.form_submit_button("開始檢索並回答")

    if submitted:
        st.markdown("---")

        with st.spinner("正在檢索投影片..."):
            answer_stream, results = answer_question_for_ui(
                query=query,
                index_dir=INDEX_DIR,
                model=model,
                top_k=top_k,
                keyword_k=keyword_k,
                embedding_k=embedding_k,
                rrf_k=RRF_K,
            )

        st.subheader("答案")

        answer_placeholder = st.empty()
        full_answer = ""

        for chunk in answer_stream:
            full_answer += chunk
            answer_placeholder.markdown(full_answer)

        st.subheader("Retrieved Slides")

        for item in results:
            with st.container(border=True):
                show_retrieved_slide(item, INDEX_DIR)


if __name__ == "__main__":
    main()