import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.llms import HuggingFaceEndpoint

# -----------------------------
# Step 1: Load PDF text (assumed pre-extracted)
# In practice, you could extract text from PDFs using a library such as PyMuPDF.
def load_pdf_text(file_path):
    # For this example, assume text files contain the extracted PDF content.
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Example: Two travel insurance product documents
pdf_text1 = load_pdf_text("travel_product1.txt")
pdf_text2 = load_pdf_text("travel_product2.txt")
all_texts = [pdf_text1, pdf_text2]

# -----------------------------
# Step 2: Split the text into chunks for retrieval
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for text in all_texts:
    chunks.extend(splitter.split_text(text))
print(f"Total chunks created: {len(chunks)}")

# -----------------------------
# Step 3: Create embeddings for each chunk
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Step 4: Build a FAISS index from the chunks
faiss_index = FAISS.from_texts(chunks, embeddings)

# -----------------------------
# Step 5: Set up the LLaMA-3-70B LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-70B",
    temperature=0.3,
    top_p=0.9,
    task="text-generation",
    model_kwargs={"max_length": 1024}
)

# -----------------------------
# Step 6: Define a prompt template to force reasoning using only the retrieved context
template = """
You are an expert insurance advisor. Based solely on the following information extracted from our travel insurance product documents:

{context}

Answer the following scenario-based question by comparing available travel insurance products and plans. Consider differences in coverage, benefits, add-ons, and pricing. If the answer is not clearly available from the context, respond with "I do not have enough information to answer this question."

Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# -----------------------------
# Step 7: Create a function to answer questions using RAG
def answer_question(question: str) -> str:
    # Retrieve the top 5 relevant chunks from the FAISS index based on the question
    docs = faiss_index.similarity_search(question, k=5)
    # Combine the retrieved chunks into one context string
    context = "\n\n".join([doc.page_content for doc in docs])
    # Format the prompt with the retrieved context and the question
    formatted_prompt = prompt.format(context=context, question=question)
    # Call the LLaMA endpoint with the formatted prompt
    answer = llm(formatted_prompt)
    return answer

# -----------------------------
# Example usage
if __name__ == "__main__":
    sample_question = "If my baggage is damaged while in the custody of the airline, but I don’t report it immediately, can I still file a claim?"
    response = answer_question(sample_question)
    print("Answer:", response)
