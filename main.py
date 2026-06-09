from __future__ import annotations

import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from slide_retriever import (
    encode_image,
    get_hybrid_results,
    make_answer_context,
)

# 載入 .env
load_dotenv()

# ==========================
# 參數設定
# ==========================

INDEX_DIR = "index"
MODEL = "gpt-4.1-mini"

TOP_K = 5
KEYWORD_K = 30
EMBEDDING_K = 30
RRF_K = 60

INPUT_CSV = "NLP期末專題_測資範例.csv"
OUTPUT_CSV = "output_v5.csv"


# ==========================
# 呼叫 OpenAI 取得答案
# ==========================

def generate_answer(
    content: list[dict],
    model: str = MODEL,
) -> tuple[str, int]:

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

    answer = response.output_text

    usage = response.usage
    total_tokens = usage.total_tokens if usage else 0

    return answer, total_tokens


# ==========================
# 單一題目回答
# ==========================

def answer_question(
    cache: IndexCache,
    query: str,
    index_dir: str = INDEX_DIR,
    model: str = MODEL,
    top_k: int = TOP_K,
    keyword_k: int = KEYWORD_K,
    embedding_k: int = EMBEDDING_K,
    rrf_k: int = RRF_K,
) -> str:

    # Step 1. 檢索相關投影片
    results = get_hybrid_results(
    cache=cache,
    query=query,
    top_k=top_k,
    keyword_k=keyword_k,
    embedding_k=embedding_k,
    rrf_k=rrf_k,
)

    # Step 2. 建立文字 context
    context = make_answer_context(results)

    # Step 3. Prompt
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
9. 請務必使用與 User Question 完全相同的語言回答。

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

    # Step 4. 建立 content
    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    image_dir = Path(index_dir) / "slide_images"

    # Step 5. 加入圖片
    for item in results:

        image_path = image_dir / f'{item["slide_id"]}.png'

        if image_path.exists():

            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f'Image for Slide {item["rank"]}: '
                        f'{item["filename"]}, '
                        f'p.{item["page"]}, '
                        f'title: {item["title"]}'
                    ),
                }
            )

            content.append(
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:image/png;base64,"
                        f"{encode_image(image_path)}"
                    ),
                }
            )

    # Step 6. 呼叫 GPT
    answer, total_tokens = generate_answer(content, model)

    return answer, total_tokens


# ==========================
# 主程式
# ==========================

def main():
    from slide_retriever import IndexCache

    print("Loading index...")

    cache = IndexCache(INDEX_DIR)

    print("Index loaded.")
    print("讀取 CSV...")

    # CSV 沒有欄位名稱
    df = pd.read_csv(INPUT_CSV, header=None)

    questions = df.iloc[:, 0]

    answers = []
    token_list = []

    total_tokens = 0

    total = len(df)

    for idx, question in enumerate(questions, start=1):

        print("=" * 60)
        print(f"[{idx}/{total}]")

        if pd.isna(question):

            print("空白題目，跳過")

            answers.append("")
            token_list.append(0)

            continue

        question = str(question).strip()

        print("問題：", question)

        try:
            # 正確解包回傳的 Tuple
            answer, tokens = answer_question(
                cache=cache,
                query=question,
            )

            print(f"完成，使用 {tokens} tokens")
            print(f"答案: {answer} ")

        except Exception as e:

            answer = f"Error: {str(e)}"
            tokens = 0

            print("失敗：", e)

        answers.append(answer)
        token_list.append(tokens)

        total_tokens += tokens

    # 第一欄保留原問題
    # 第二欄答案
    # 第三欄 token
    output_df = pd.DataFrame({
        "Question": questions,
        "Answer": answers,
        "Tokens": token_list,
    })

    print("輸出 CSV...")

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"完成！已儲存至 {OUTPUT_CSV}")
    print(f"總 Token 使用量：{total_tokens}")


# ==========================
# 執行
# ==========================

if __name__ == "__main__":
    main()