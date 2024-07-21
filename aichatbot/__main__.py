import argparse
import importlib.resources as pkg_resources
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from operator import itemgetter
import os


# ANSI escape sequences for colors
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_or_create_vector_db(pdf_resource_path, embedding_dir):
    if os.path.exists(embedding_dir) and os.path.isdir(embedding_dir):
        print(f"Loading embeddings from {embedding_dir}")
        return Chroma(
            persist_directory=embedding_dir,
            embedding_function=OllamaEmbeddings(
                model="nomic-embed-text", show_progress=True
            ),
            collection_name="local-rag",
        )
    else:
        print("Creating new embeddings...")
        chunks = load_and_split_pdfs(pdf_resource_path)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
            persist_directory=embedding_dir,
            collection_name="local-rag",
        )
        vector_db.persist()
        print(f"Embeddings saved to {embedding_dir}")
        return vector_db


def create_vector_db(chunks, embedding_function):
    return Chroma.from_documents(
        documents=chunks, embedding=embedding_function, collection_name="local-rag"
    )


def load_and_split_pdfs(pdf_resource_path):
    pdf_dir = pkg_resources.files(pdf_resource_path)
    pdf_files = [entry for entry in pdf_dir.iterdir() if entry.suffix == ".pdf"]
    all_data = []

    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}")
        local_path = str(pdf_file)
        try:
            loader = UnstructuredPDFLoader(file_path=local_path)
            data = loader.load()
            all_data.extend(data)
        except Exception as e:
            print(f"Could not load file {local_path}: {e}")

    if not all_data:
        print("No data loaded from PDF files.")
        exit(1)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    return text_splitter.split_documents(all_data)


from operator import itemgetter


def main(pdf_resource_path, model_name, input_question, embedding_dir):
    vector_db = get_or_create_vector_db(pdf_resource_path, embedding_dir)

    llm = ChatOllama(model=model_name)

    # Enhance the retriever without score_threshold
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},  # Increase k to get more results initially
    )

    # Improve the prompt
    template = """Use ONLY the following pieces of context to answer the question at the end.
    If you don't know the answer based solely on this context, say "I don't have enough information to answer this question."
    Always cite the specific part of the context you used to answer the question.

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    # Get the answer
    result = qa_chain.invoke({"query": input_question})
    answer = result["result"]
    source_docs = result["source_documents"]

    # Post-processing step to filter source documents
    filtered_docs = []
    for doc in source_docs:
        if hasattr(doc, "metadata") and "score" in doc.metadata:
            if doc.metadata["score"] > 0.5:  # Adjust this threshold as needed
                filtered_docs.append(doc)
        else:
            filtered_docs.append(doc)  # Include docs without scores

    # Sort filtered docs by score if available
    filtered_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)

    if not filtered_docs:
        print(
            f"{Colors.WARNING}No relevant information found in the context. The model may not have accurate information to answer this question.{Colors.ENDC}"
        )
    else:
        print(f"{Colors.OKGREEN}{answer}{Colors.ENDC}")

    # Print the filtered and sorted source documents
    print("\nSource Documents:")
    for doc in filtered_docs:
        score = doc.metadata.get("score", "N/A")
        print(f"{Colors.OKCYAN}Score: {score}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}{doc.page_content[:200]}...{Colors.ENDC}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PDF and query using LangChain with saved embeddings."
    )
    parser.add_argument(
        "--pdf_resource_path",
        type=str,
        required=True,
        help="The resource path of the PDF files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the LLM model to use.",
    )
    parser.add_argument(
        "--input_question", type=str, required=True, help="The input question to ask."
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="The directory to save/load embeddings.",
    )

    args = parser.parse_args()
    main(
        args.pdf_resource_path,
        args.model_name,
        args.input_question,
        args.embedding_dir,
    )
