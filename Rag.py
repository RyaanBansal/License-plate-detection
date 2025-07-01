from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
print("GROQ API Key Loaded:", bool(groq_api_key))

def rag_pipeline(query):
        # 1. Load Documents
    print("🔹 Loading PDF document...")
    if not os.path.exists("License_Plates.pdf"):
        raise FileNotFoundError("License_Plates.pdf not found.")
    loader = PyPDFLoader("License_Plates.pdf")
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document(s).")

        # 2. Split Text
    print("🔹 Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=600)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")

        # 3. Embeddings
    print("🔹 Creating HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 4. Vector Store
        
    try:
        print("🔹 Creating FAISS vector store...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        print("✅ FAISS vector store created.")
    except Exception as e:
        print("❌ Error creating FAISS vector store:", e)
        return "Error in vector store creation."


        # 5. Retrieval
    print("🔹 Retrieving relevant documents...")
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(query)
    print(f"✅ Retrieved {len(retrieved_docs)} relevant document(s).")

    if not retrieved_docs:
        print("⚠️ No relevant documents found. Try a different query.")
        return "No relevant information found."

        # 6. Generation using Groq
    print("🔹 Generating answer using Groq LLM...")
    llm = ChatGroq(model_name="llama3-70b-8192", temperature= 0.2)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
    You are an expert at analysing vehicle registration documents and number plates.
    Given the context and the presented question, formulate a response by going through the context and accurately determine the appropriate answer.
    Context: {context}
    
    Question: {query}
    Present the answer in bullet points, make sure that all the requested type of number plates are displayed."""
    answer = llm.invoke(prompt)

    print("✅ Answer generated.")
    return getattr(answer, "content", answer)

# Run the pipeline
question = """Provide all the number plates from India present in the data. Follow the standard format for an Indian number plate"""
print(f"\n🔍 Question: {question}")
answer = rag_pipeline(question)
print(f"\n📝 Answer: {answer}")
