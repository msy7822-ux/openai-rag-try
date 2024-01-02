import utils
import retriever as Retriever
import rag
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage import InMemoryStore


def main():
    # PDFの分解（テキスト、テーブル、画像）
    tables, texts = utils.process_pdf("attention_is_all_you_need.pdf")
    # テーブルのサマリ作成
    table_summaries = utils.summarize_tables(tables)
    # 画像のサマリ作成
    img_base64_list, image_summaries = utils.summarize_images()

    # Multivector Retrieverの作成
    vectorstore = Chroma(
        collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings()
    )
    docstore = InMemoryStore()
    retriever = Retriever.create_vectorstore(
        vectorstore,
        docstore,
        texts,
        table_summaries,
        image_summaries,
        tables,
        img_base64_list,
    )

    questions = ["Attention is all you needとは何か？", "Attention is all you needの著者は？"]

    for query in questions:
        print(f"Q: {query}")
        result = rag.rag_application(query, retriever)
        print(f"A: {result}\n\n")
        print("----------------------------------")


# if __name__ == "__main__":
main()
