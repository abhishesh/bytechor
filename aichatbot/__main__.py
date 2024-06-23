import argparse
import importlib.resources as pkg_resources
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# ANSI escape sequences for colors
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"  # Resets all colors and styles
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def main(pdf_resource_path, model_name, input_question):
    # Get all PDF files in the resource path
    pdf_dir = pkg_resources.files(pdf_resource_path)
    pdf_files = [entry for entry in pdf_dir.iterdir() if entry.suffix == ".pdf"]
    all_data = []

    for pdf_file in pdf_files:
        print(f"loading {pdf_file}")
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

    # Split and chunk the loaded data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_data)

    # Add chunks to the vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag",
    )

    # Define the LLM model
    llm = ChatOllama(model=model_name)

    # Define the query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Initialize the retriever with the LLM and query prompt
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    # Define the RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Define the chain for processing the input and getting the answer
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain with an input question and print the answer
    answer = chain.invoke(input_question)
    print(f"{Colors.OKGREEN}{answer}{Colors.ENDC}")

    # Delete all collections in the vector database
    vector_db.delete_collection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PDF and query using LangChain."
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

    args = parser.parse_args()
    main(args.pdf_resource_path, args.model_name, args.input_question)
