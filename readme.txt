1.先設定api key
setx OPENAI_API_KEY "你的API金鑰"

2.設定完後cmd要關掉重開

3.cd C:\Users\a30099\Desktop\資工二技\四下\自然語言處理\nlp_slide_retriever

4.streamlit run app.py


5.加入新的PDF 
5-1.caption:
python slide_retriever.py caption ^
--pdf_dir data/pdfs ^
--index_dir index ^
--model gpt-4.1

5-2.build
python slide_retriever.py build ^
--pdf_dir data/pdfs ^
--output_dir index ^
--min_df 1 ^
--top_features 100000

5-3.重新建立 embedding index
python slide_retriever.py embed ^
--index_dir index ^
--model text-embedding-3-large ^
--batch_size 64

5-4.測試新 PDF 是否有進系統
python slide_retriever.py answer ^
--index_dir index ^
--query "你的新問題" ^
--top_k 5 ^
--keyword_k 30 ^
--embedding_k 30 ^
--model gpt-4.1

6.執行回答

python batch_answer_csv_fast.py ^
--input_csv NLP_QA.csv ^
--output_csv NLP_QA_答案_auto.csv ^
--index_dir index ^
--model gpt-4.1-mini ^
--top_k 3 ^
--neighbor 1 ^
--image_mode auto ^
--image_limit 1 ^
--max_output_tokens 100 ^
--sleep_seconds 0.5
