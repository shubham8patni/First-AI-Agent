import os
import json
import fitz  # PyMuPDF
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
# from langchain import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.tools import Tool, tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.retrievers.multi_query import MultiQueryRetriever
from tavily import TavilyClient
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)
# tavily = TavilyClient(api_key=os.getenv("TAVILY_KEY"))

travel_data_domestic = [
    {
        "destination": "Jakarta Barat",
        "startDate": "01/09/2025",
        "endDate": "31/01/2026",
        "event": "It's best to avoid traveling to the Indonesian capital Jakarta during the rainy season from September to January, as the weather tends to be humid, and some areas may face flooding. Due to flooding there can be travel delays and loss of hotel reservations as well."
    },
    {
        "destination": "Bali",
        "startDate": "01/12/2025",
        "endDate": "31/01/2026",
        "event": "Those looking for a quintessential Bali vacation with plenty of sunshine and outdoor activities will want to avoid the rainy season (especially during the wettest months, December and January)."
    },
    {
        "destination": "Bintan",
        "startDate": "01/11/2025",
        "endDate": "31/12/2025",
        "event": "Rain on a tropical island like Bintan tends to occur sporadically through the year, though you have the highest chance of encountering heavy rains during the months of November and December when the monsoon season begins. There might be travel delays or cancellations or loss of hotel reservations during this month."
    }
]

travel_data_intenational =  [
        {
            "destination": "India",
            "startDate": "01/01/2025",
            "endDate": "30/04/2025",
            "event": "There will be Maha kumbh religious event in india from January to April. There will be too much travellers movement during this time which might result in travel delays and baggage loss or delaya. Hotel reservations might also get cancelled which might result in inconveience. Both train and air travel is costly durign this time."
        },
        {
            "destination": "Japan",
            "startDate": "01/07/2025",
            "endDate": "31/08/2025",
            "event": "This is time of extreme weather in Japan from July to August. Due to extreme weather there might be delays in baggage, flight getting cancelled. Loss of hotel reservations due to weather conditions or delays on travel will be normal."
        },
        {
            "destination": "Philippines",
            "startDate": "01/07/2025",
            "endDate": "30/09/2025",
            "event": "This is period of typhoons weather in philippines from July to September. Due to extreme weather there might be delays in baggage, flight getting cancelled. Loss of hotel reservations due to weather conditions or delays on travel will be normal."
        },
        {
            "destination": "Russia",
            "startDate": "01/01/2025",
            "endDate": "30/03/2025",
            "event": "This is period of extreme cold weather in russia from January to March. Due to extreme weather there might be delays in baggage, flight getting cancelled. Loss of hotel reservations due to weather conditions or delays on travel will be normal."
        }
]

########################################
# Document Loading and Preprocessing
########################################

def load_pdf_documents(pdf_paths):
    """Load PDFs and return a list of Document objects."""
    documents = []
    for file_path in pdf_paths:
        try:
            with fitz.open(file_path) as doc:
                text = "\n".join([page.get_text() for page in doc])
                documents.append(Document(page_content=text, metadata={"source": file_path}))
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
    return documents

def load_json_documents(json_path):
    """Load JSON file and convert its content into a Document object.
       Assumes the JSON contains structured policy data."""
    documents = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Here we simply dump the entire JSON as text.
            content = json.dumps(data, indent=2)
            documents.append(Document(page_content=content, metadata={"source": json_path}))
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
    return documents

# Define your source file paths
PDF_PATHS = [
    # "Zurich_sompo_Domestic_Travel_Insurance_final.pdf",
    # "Zurich_Policy_Wording.pdf",
    # "Policy_Wording_SOMPO_Domestic.pdf",
    # "Zurich_sompo_International_ASEAN.pdf",
    # "sompodom_merged.pdf"
]
JSON_PATH = "zurich_insurance_jon.json"

# Load documents from both PDFs and JSON
pdf_docs = load_pdf_documents(PDF_PATHS)
json_docs = load_json_documents(JSON_PATH)
all_documents = pdf_docs + json_docs

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(all_documents)

########################################
# Build FAISS Vector Store
########################################

# Initialize embedding model (using a Sentence Transformer)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the FAISS index from document chunks
db = FAISS.from_documents(chunks, embeddings)

########################################
# Retrieval and Answer Generation Functions
########################################

# Define a prompt template for retrieval-augmented generation
prompt_template = """
You are an AI insurance assistant. Given the following retrieved policy information:

{context}

Answer the user's question fully. If multiple plans or coverages apply, include all relevant details.
If the answer cannot be found in the provided context, respond with "I do not have enough information to answer this question."

Now answer the following question:
User: {question}
Retrieved Info:
{context}

Response:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

def get_relevant_context(query: str, retriever, k: int = 4) -> str:
    """Fetch top-k relevant document chunks for the given query."""
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs[:k]])
    return context if context else "No relevant policy data found."

def retrieve_and_answer(query: str) -> str:
    """Retrieve policy data from FAISS and generate an answer using LLaMA."""
    # print("\n========== PRE Retrieved Context ==========\n",query)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    context = get_relevant_context(query, retriever)
    print("\n========== Retrieved Context ==========\n", context)
    if context == "No relevant policy data found.":
        return "I do not have enough information to answer this question."
    formatted_prompt = prompt.format(context=context, question=query)
    response = llm(formatted_prompt)
    return response

########################################
# LLM and Agent Configuration
########################################

# Configure LLaMA models via HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",  # or use another repo id for LLaMA 3-70B if available
    temperature=0.4,
    top_p=0.7,
    task="text-generation",
    model_kwargs={"max_length": 5000}
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-70B",
    temperature=0.3,
    top_p=0.6,
    task="text-generation",
    model_kwargs={"max_length": 4096}
)

# Set up Multi-Query Retriever for improved search
multi_query_retriever = MultiQueryRetriever.from_llm(llm=llm, retriever=db.as_retriever(search_kwargs={"k": 4}))

# Memory-enabled conversational retrieval chain
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer", max_token_limit=1000)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=multi_query_retriever,
    verbose=True,
    return_source_documents=False,
    output_key="answer"
)

def retrieve_insurance_info(query: str) -> str:
    """Retrieve insurance information using the conversational retrieval chain."""
    print("\n========== PRE Retrieved Context ==========\n",query)
    response_data = qa_chain.invoke({"query": query})
    print("\n========== Retrieved Context Response Data ==========\n", response_data)
    final_answer = response_data["answer"]
    print("\n========== Retrieved Context ==========\n", final_answer)
    if not final_answer or "I do not have enough information" in final_answer:
        return "I do not have enough information to answer this question."
    return final_answer

insurance_tool = Tool(
    name="InsuranceRetrieval",
    func=retrieve_insurance_info,
    description="Use this tool to answer insurance policy questions."
)

# Strict prompt with hallucination prevention (for agent reasoning)
strict_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template = """
    You are an AI insurance assistant. Use only the retrieved policy information below to answer the user's question. 
    Your response must strictly follow the format below:

    1. A **Thought** section that briefly explains your reasoning.
    2. An **Action** section that provides the final answer.

    If the answer cannot be found in the provided context, respond with:
    "I do not have enough information to answer this question."

    Example Format:
    Thought: [Your reasoning...]
    Action: [Your final answer.]

    Now, answer the following:
    User: {question}
    Retrieved Info:
    {context}

    Response:
    """
# You are an AI insurance assistant. Given the following retrieved policy information:

# {context}

# Answer the user's question fully. If multiple plans or coverages apply, include all relevant details.
# If the answer cannot be found in the provided context, respond with "I do not have enough information to answer this question."

# Now answer the following question:
# User: {question}
# Retrieved Info:
# {context}

# Response:
# """
)

agent = initialize_agent(
    tools=[insurance_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=20,
    max_execution_time=45,
    handle_parsing_errors=True,
    combine_docs_chain=LLMChain(llm=llm, prompt=strict_prompt),
    verbose=True
)
print([tool.name for tool in [insurance_tool]])  # Debugging step
########################################
# API Endpoints
########################################

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    try:
        response = agent.run(user_question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/recommend_mock", methods=["POST"])
def recommend_mock():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON or empty request body"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    destination = data.get("destination", "")
    category = data.get("category", "")
    product_name = data.get("product", "")
    travel_data = {}

    if category == "International":
        template =  """You are an AI travel insurance assistant. Your goal is to recommend the best travel insurance product (Zurich or Sompo) based on the latest travel insights.
        
        ### Latest Travel Insights:
        - {tavily_summary}
        
        Provide a recommendation for someone visiting {destination}, ensuring:\n1. Select the best insurance provider (Zurich or Sompo) based on coverage.\n2. List the plan name and key benefits.\n3. Explain why this plan is best based on risks and travel conditions.
        
        Recommended Insurance Plan:\n- Provider: Zurich / Sompo\n- Plan Name: [Plan name]\n- Key Benefits: \n  - [Coverage 1]\n  - [Coverage 2]\n  - [Coverage 3]\n- Why This Plan? [Explain reasoning]\n\nAdditional Add-On Recommendations:\n- [Add-on 1]: [Why needed]\n- [Add-on 2]: [Why needed]\n\nEnsure the response is well-organized, professional, and easy to understand, written strictly in English."""
        for travel in travel_data_intenational:
            if travel["destination"] == destination:
                travel_data["answer"] = travel["event"]
    elif category == "Domestic":
        template = """You are an AI travel insurance assistant. Your goal is to recommend the best travel insurance product (Zurich or Sompo) based on the latest travel insights.
        
        ### Latest Travel Insights:
        - {tavily_summary}
        
        Provide a recommendation for someone visiting {destination}, ensuring:\n1. Select the best insurance provider (Zurich or Sompo) based on coverage.\n2. List the plan name and key benefits.\n3. Explain why this plan is best based on risks and travel conditions.
        
        Recommended Insurance Plan:\n- Provider: Zurich / Sompo\n- Plan Name: [Plan name]\n- Key Benefits: \n  - [Coverage 1]\n  - [Coverage 2]\n  - [Coverage 3]\n- Why This Plan? [Explain reasoning]\n\nAdditional Add-On Recommendations:\n- [Add-on 1]: [Why needed]\n- [Add-on 2]: [Why needed]\n\nEnsure the response is well-organized, professional, and easy to understand, written strictly in English."""
        for travel in travel_data_domestic:
            if travel["destination"] == destination:
                travel_data["answer"] = travel["event"]
    else:
        return jsonify({"error": "Invalid category provided"}), 400

    if not travel_data or "answer" not in travel_data:
       return jsonify({"error": "Failed to fetch travel data"}), 500
    
    prompt_mock = PromptTemplate(input_variables=["destination", "tavily_summary", "product_name"], template=template)
    chain = LLMChain(llm=llm2, prompt=prompt_mock)
    recommendation = chain.run(destination=destination, tavily_summary=travel_data["answer"], product_name=product_name)
    return jsonify({"recommendation": recommendation})

@app.route("/recommend_addon_mock", methods=["POST"])
def recommend_addon_mock():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON or empty request body"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    destination = data.get("destination", "")
    category = data.get("category", "")
    product_name = data.get("product", "")
    travel_data = {}
    if category == "International":
        template = """You are an AI travel insurance assistant. Your goal is to recommend the best travel insurance Add-on/additional benefits for ({product_name}) based on the latest travel insights.
        
        ### Latest Travel Insights:
        - {tavily_summary}
        
        Provide a recommendation for someone visiting {destination}, ensuring:\n1. Select the best Add-on/additional benefits ({product_name}) based on coverage.\n2. List the add-on name and key benefits.\n3. Explain why this add-on/additional benefit is best based on risks and travel conditions.
        
        Recommended Insurance Plan:\n- Provider: {product_name}\n- Plan Name: [Plan name]\n\nAdd-On Recommendations:\n- [Add-on 1]: [Why needed]\n- [Add-on 2]: [Why needed]\n- Why These add-ons? [Explain reasoning]\n\nEnsure the response is in plain text format, well-organized, professional, and easy to understand, written strictly in English."""
        for travel in travel_data_intenational:
            if travel["destination"] == destination:
                travel_data["answer"] = travel["event"]
    elif category == "Domestic":
        template = """You are an AI travel insurance assistant. Your goal is to recommend the best travel insurance Add-on/additional benefits for ({product_name}) based on the latest travel insights.
        
        ### Latest Travel Insights:
        - {tavily_summary}
        
        Provide a recommendation for someone visiting {destination}, ensuring:\n1. Select the best Add-on/additional benefits ({product_name}) based on coverage.\n2. List the add-on name and key benefits.\n3. Explain why this add-on/additional benefit is best based on risks and travel conditions.
        
        Recommended Insurance Plan:\n- Provider: {product_name}\n- Plan Name: [Plan name]\n\nAdd-On Recommendations:\n- [Add-on 1]: [Why needed]\n- [Add-on 2]: [Why needed]\n- Why These add-ons? [Explain reasoning]\n\nEnsure the response is in plain text format, well-organized, professional, and easy to understand, written strictly in English."""
        for travel in travel_data_domestic:
            if travel["destination"] == destination:
                travel_data["answer"] = travel["event"]
    else:
        return jsonify({"error": "Invalid category provided"}), 400

    if not travel_data or "answer" not in travel_data:
       return jsonify({"error": "Failed to fetch travel data"}), 500
    
    prompt_addon = PromptTemplate(input_variables=["destination", "tavily_summary", "product_name"], template=template)
    chain = LLMChain(llm=llm2, prompt=prompt_addon)
    recommendation = chain.run(destination=destination, tavily_summary=travel_data["answer"], product_name=product_name)
    return jsonify({"recommendation": recommendation})

########################################
# Run Flask App
########################################
if __name__ == "__main__":
    app.run(debug=True)
