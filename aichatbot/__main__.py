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


# Define the local path for the PDF file
pdf_path = pkg_resources.files("aichatbot.resources").joinpath("bitcoin.pdf")
local_path = str(pdf_path)
try:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
except Exception as e:
    print("could not load file")
    exit(1)

# Split and chunk the loaded data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Add chunks to the vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag",
)

# Define the LLM model
local_model = "mistral"
llm = ChatOllama(model=local_model)

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
input_question = "Whats is proof of work?"
answer = chain.invoke(input_question)


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


# Print Answer
print(f"{Colors.OKGREEN}{answer}{Colors.ENDC}")

# Delete all collections in the vector database
vector_db.delete_collection()
