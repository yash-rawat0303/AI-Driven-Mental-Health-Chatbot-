import os
import logging
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.WARNING)

def initialize_llm():
    """Initialize the Groq LLM."""
    return ChatGroq(
        temperature=0,
        groq_api_key="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",  # Replace with your key or use os.getenv("GROQ_API_KEY")
        model_name="llama-3.3-70b-versatile"
    )

def create_vector_db(pdf_path: str = "./data/", db_dir: str = "./chroma_db"):
    """Create or load a Chroma vector database from PDFs."""
    os.makedirs(db_dir, exist_ok=True)

    # Load documents
    if os.path.isdir(pdf_path):
        loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
    elif os.path.isfile(pdf_path) and pdf_path.endswith(".pdf"):
        loader = PyPDFLoader(pdf_path)
    else:
        raise ValueError(f"Invalid path: {pdf_path}. Must be a directory or a PDF file.")

    docs = loader.load()
    if not docs:
        raise FileNotFoundError(f"No documents found at {pdf_path}")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create embeddings and vector DB
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedder, persist_directory=db_dir)
    db.persist()
    print(f"Chroma DB created and saved at: {db_dir}")
    return db

def setup_qa_chain(db: Chroma, llm):
    """Set up the RetrievalQA chain with a mental health-focused prompt."""
    prompt = PromptTemplate(
        template=(
            "You are a compassionate mental health chatbot. Respond thoughtfully based on the context.\n\n"
            "Context:\n{context}\n\n"
            "User: {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        chain_type_kwargs={"prompt": prompt}
    )

def run_cli(qa_chain):
    """Run the chatbot in CLI mode."""
    print("Chatbot is ready! Type your question or 'exit' to quit.")
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye! Take care.")
            break
        try:
            answer = qa_chain.run(query)
            print(f"Bot: {answer}")
        except Exception as e:
            print(f"Error: {e}")

def chatbot_response(user_input, history=[]):
    """Handle Gradio chatbot responses."""
    if not user_input.strip():
        return "Please provide a valid input.", history

    try:
        response = qa_chain.run(user_input)
        # Basic formatting (remove extra brackets/quotes if present)
        response = re.sub(r'\$.*?\$|\$.*?\$|\{.*?\}|\'.*?\'|".*?"', '', response)
        response = re.sub(r'\s+', ' ', response).strip()
        return response, history
    except Exception as e:
        return f"An error occurred: {e}", history

# Main setup
if __name__ == "__main__":
    llm = initialize_llm()
    pdf_path = "./data/"  # Adjust to your local PDF folder or file path
    db_dir = "./chroma_db"

    # Load or create vector DB
    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        print("Building vector DB...")
        db = create_vector_db(pdf_path, db_dir)
    else:
        print("Loading existing Chroma DB...")
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=db_dir, embedding_function=embedder)

    qa_chain = setup_qa_chain(db, llm)

    # Choose mode: "cli" for command-line, "gradio" for web interface
    mode = "cli"  # Change to "gradio" for web UI

    if mode == "cli":
        run_cli(qa_chain)
    elif mode == "gradio":
        with gr.Interface(
            fn=chatbot_response,
            inputs="text",
            outputs="text",
            title="Mental Health Chatbot"
        ) as app:
            app.launch()  # Launches locally; access at http://127.0.0.1:7860
    else:
        print("Invalid mode. Set 'mode' to 'cli' or 'gradio'.")