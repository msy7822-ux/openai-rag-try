# ## URL: https://qiita.com/t_serizawa/items/a2ced4441da714b3076f

# import pandas as pd
# from datasets import load_dataset

# from langchain import PromptTemplate, LLMChain
# from langchain.llms import GPT4All
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain import PromptTemplate
# from langchain.document_loaders import TextLoader
# from langchain import PromptTemplate
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from langchain.document_loaders import TextLoader
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.embeddings import HuggingFaceEmbeddings

# dataset = load_dataset("medical_dialog", "processed.en")
# df = pd.DataFrame(dataset["train"])

# dialog = []

# medical_data_file = "./qiita_rag/medical_data.txt"

# # 患者と医者の発言をそれぞれ抽出した後、順にリストに格納
# patient, doctor = zip(*df["utterances"])
# for i in range(len(patient)):
#     dialog.append(patient[i])
#     dialog.append(doctor[i])

# df_dialog = pd.DataFrame({"dialog": dialog})

# # 成形終了したデータセットを保存
# df_dialog.to_csv(medical_data_file, sep=" ", index=False)

# loader = TextLoader(medical_data_file, encoding="utf-8")
# index = VectorstoreIndexCreator(embedding=HuggingFaceEmbeddings()).from_loaders(
#     [loader]
# )

# # llm_path = "./model/ggml-gpt4all-j-v1.3-groovy.bin"  # replace with your desired local file path
# llm_path = "gpt-3.5-turbo-instruct"  # replace with your desired local file path
# callbacks = [StreamingStdOutCallbackHandler()]
# llm = GPT4All(model=llm_path, callbacks=callbacks, verbose=True, backend="gptj")

# results = index.vectorstore.similarity_search(
#     "what is the solution for soar throat", k=4
# )
# context = "\n".join([document.page_content for document in results])
# print(f"{context}")
