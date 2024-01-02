from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks import get_openai_callback


def rag_application(question, retriever):
    docs = retriever.get_relevant_documents(question)
    docs_by_type = split_image_text_types(docs)

    # テーブルの取得と表示
    if len(docs_by_type["texts"]):
        for doc_id in docs_by_type["texts"]:
            doc = retriever.docstore.mget([doc_id])
            try:
                doc_html = convert_html(doc)
                display(HTML(doc_html))
            except Exception as e:
                print(doc)

    # 画像の取得と表示
    if len(docs_by_type["images"]):
        plt_img_base64(docs_by_type["images"][0])

    model = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024, temperature=0)
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(generate_prompt)
        | model
        # | StrOutputParser()
    )
    answer = chain.invoke(question)

    return answer


from base64 import b64decode
from IPython.display import display, HTML
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage


def split_image_text_types(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}


def plt_img_base64(img_base64):
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(image_html))


def convert_html(element):
    input_text = str(element)
    prompt_text = f"""
    回答例にならって、テキストをHTMLテーブル形式に変換してください:
    
    テキスト：
    {input_text}
    
    回答例：
    項目 果物(kg) お菓子(kg) ナッツ(kg) 飲み物(L) 予想 45 20 15 60 実績 50 25 10 80 
    差(実績-予想) 5 5 -5 -20
    →
    <table>
      <tr>
        <th>項目</th>
        <th>果物(kg)</th>
        <th>お菓子(kg)</th>
     ・・・
      </tr>
    </table>
    """
    message = HumanMessage(content=[{"type": "text", "text": prompt_text}])
    model = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
    response = model.invoke([message])
    return response.content


def generate_prompt(dict):
    format_texts = "\n\n".join(dict["context"]["texts"])
    prompt_text = f"""
        以下の質問に基づいて回答を生成してください。
        回答は、提供された追加情報を考慮してください。
    
        質問: {dict["question"]}

        追加情報: {format_texts}
        """
    message_content = [{"type": "text", "text": prompt_text}]

    # 画像が存在する場合のみ画像URLを追加
    if dict["context"]["images"]:
        image_url = f"data:image/jpeg;base64,{dict['context']['images'][0]}"
        message_content.append({"type": "image_url", "image_url": {"url": image_url}})

    return [HumanMessage(content=message_content)]
