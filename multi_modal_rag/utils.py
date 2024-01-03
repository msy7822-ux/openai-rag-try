## URL: https://qiita.com/mashmoeiar11/items/c89ae3f5084680676611

import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

from unstructured.partition.pdf import partition_pdf

path = "/Users/msy/ai/openai-tutorial/multi_modal_rag/"


def delete_small_files(directory_path, max_size_kb=40):
    max_size_bytes = max_size_kb * 1024
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            if file_size <= max_size_bytes:
                os.remove(file_path)
                print(f"Deleted: {file_path}")


def process_pdf(file):
    # Get Elements
    raw_pdf_elements = partition_pdf(
        filename=path + file,
        languages=["jpn", "eng"],
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path + "public/output/",
    )

    # delete small size files
    delete_small_files(path + "public/output/")

    # Get tables and texts
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    return tables, texts


import os
import base64
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage


def summarize_images():
    img_base64_list = []
    image_summaries = []
    img_prompt = "画像を日本語で詳細に説明してください。"

    for img_file in sorted(os.listdir(path + "public/output/")):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path + "public/output/", img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, img_prompt))

    return img_base64_list, image_summaries


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )

    return msg.content


from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate


def summarize_tables(tables):
    table_prompt = """あなたはテーブルの内容を説明する役割をもっています。
    (1) 何に関してのテーブルなのか、
    (2) テーブルの詳細内容と考察
    に関して日本語で説明してください。

    テーブル（テキスト）: 
    {element}
    """
    prompt = ChatPromptTemplate.from_template(table_prompt)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    assert len(table_summaries) == len(
        tables
    ), "Summary tables and original tables count must be equal"

    return table_summaries
