import os
import fitz  # PyMuPDF
import re
from flask import Flask, request, jsonify
from langchain import PromptTemplate
# import langchain_core.prompts.PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.retrievers.multi_query import MultiQueryRetriever
from tavily import TavilyClient
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
tavily = TavilyClient(api_key="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")

# ========= 🔹 Load Policy Documents & Preprocess 🔹 =========
POLICY_DOCUMENT_PATHS = ["Zurich_sompo_Domestic_Travel_Insurance_final.pdf"] #["zurich_domestic_PDP.pdf"] #["sompodom_merged.pdf", "PDF_translate_Travel_zurich.pdf", "Zurich_APAC.pdf"]
policy_texts = []
travel_data_domestic = [
    {
        "destination": "jakarta",
        "startDate": "01/09/2025",
        "endDate": "31/01/2026",
        "event": "It's best to avoid traveling to the Indonesian capital Jakarta during the rainy season from September to January, as the weather tends to be humid, and some areas may face flooding. Due to flooding there can be travel delays and loss of hotel reservations as well."
    },
    {
        "destination": "bali",
        "startDate": "01/12/2025",
        "endDate": "31/01/2026",
        "event": "Those looking for a quintessential Bali vacation with plenty of sunshine and outdoor activities will want to avoid the rainy season (especially during the wettest months, December and January)."
    },
    {
        "destination": "bintan",
        "startDate": "01/11/2025",
        "endDate": "31/12/2025",
        "event": "Rain on a tropical island like Bintan tends to occur sporadically through the year, though you have the highest chance of encountering heavy rains during the months of November and December when the monsoon season begins. There might be travel delays or cancellations or loss of hotel reservations during this month."
    }
]

travel_data_intenational =  [
        {
            "destination": "india",
            "startDate": "01/01/2025",
            "endDate": "30/04/2025",
            "event": "There will be Maha kumbh religious event in india from January to April. There will be too much travellers movement during this time which might result in travel delays and baggage loss or delaya. Hotel reservations might also get cancelled which might result in inconveience. Both train and air travel is costly durign this time."
        },
        {
            "destination": "japan",
            "startDate": "01/07/2025",
            "endDate": "31/08/2025",
            "event": "This is time of extreme weather in Japan from July to August. Due to extreme weather there might be delays in baggage, flight getting cancelled. Loss of hotel reservations due to weather conditions or delays on travel will be normal."
        },
        {
            "destination": "philippine",
            "startDate": "01/07/2025",
            "endDate": "30/09/2025",
            "event": "This is period of typhoons weather in philippines from July to September. Due to extreme weather there might be delays in baggage, flight getting cancelled. Loss of hotel reservations due to weather conditions or delays on travel will be normal."
        },
        {
            "destination": "russia",
            "startDate": "01/01/2025",
            "endDate": "30/03/2025",
            "event": "This is period of extreme cold weather in russia from January to March. Due to extreme weather there might be delays in baggage, flight getting cancelled. Loss of hotel reservations due to weather conditions or delays on travel will be normal."
        }
]

try:
    for file_path in POLICY_DOCUMENT_PATHS:
        with fitz.open(file_path) as doc:
            text = "\n".join([page.get_text() for page in doc])
            policy_texts.append(text)
    policy_text = "\n\n".join(policy_texts)
except Exception as e:
    print(f"Error reading policy document: {e}")

# ========= 🔹 Chunking the Policy Text for Retrieval 🔹 =========
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
texts = text_splitter.split_text(policy_text)

# ========= 🔹 Vector Database (FAISS) 🔹 =========
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #"BAAI/bge-large-en")#
db = FAISS.from_texts(texts, embeddings)

def get_relevant_context(query: str, retriever, k: int = 3) -> str:
    """Fetch relevant policy data from FAISS for better RAG accuracy."""
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs[:k]])  # Get top K results
    return context if context else "No relevant policy data found."

# Define a prompt template
# template = """
# You are an insurance assistant. Use only the following documents to answer the question.
# If the answer cannot be found in the provided context, respond with "I do not have enough information to answer this question."

# Policy Documents:
# {context}

# Question: {question}
# """

template = """
You are an AI insurance assistant. Given the following retrieved policy information:

{context}

Answer the user's question fully. If multiple plans or coverages apply, include all relevant details.
If the answer cannot be found in the provided context, respond with "I do not have enough information to answer this question."

Example:
User: "Which plan covers accidental death? What is the coverage amount?"
Retrieved Info:
- Product Name : Plan A: Covers accidental death, $200,000 coverage.
- Product Name : Plan B: Covers accidental death, $150,000 coverage.

Response: "Product name - Plan A and Product name - Plan B both cover accidental death. Plan A provides $200,000, while Plan B provides $150,000."

Now answer the following question:
User: {question}
Retrieved Info:
{context}

Response:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

def retrieve_and_answer(query: str) -> str:
    """Retrieve policy data from FAISS and generate an AI response using Llama."""
    retriever = db.as_retriever(search_kwargs={"k": 3})  # Limit search to top 5 chunks
    context = get_relevant_context(query, retriever)

    # If no relevant data is found, return a fallback response
    if context == "No relevant policy data found.":
        return "I do not have enough information to answer this question."

    # Format the custom prompt with retrieved data
    formatted_prompt = prompt.format(context=context, question=query)

    # Send the formatted query to the LLM
    response = llm(formatted_prompt)
    return response


insurance_tool_forced_FAISS_retrival = Tool(
    name="Insurance Retrieval",
    func=retrieve_and_answer,
    description="Fetches insurance policy details strictly from stored policy documents."
)

# ========= 🔹 LLM Configuration (Meta Llama 3) 🔹 =========
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-70B",
    temperature=0.3,
    top_p=0.5,
    task="text-generation",
    model_kwargs={"max_length": 2048}  # Increase for multi-step answers
)

# ========= 🔹 Multi-Query Retriever for Improved Search 🔹 =========
multi_query_retriever = MultiQueryRetriever.from_llm(llm=llm, retriever=db.as_retriever(search_kwargs={"k": 3}))

# ========= 🔹 Memory-Enabled Conversational Retrieval 🔹 =========
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=1000)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=multi_query_retriever, memory=memory, verbose=True
)

# ========= 🔹 Wrap QA Chain in a Tool (Fix for Agents) 🔹 =========
def retrieve_insurance_info(query: str) -> str:
    """Retrieve insurance-related information using FAISS + LLM."""
    return qa_chain.run({"question": query, "chat_history": []})

insurance_tool = Tool(
    name="Insurance Retrieval",
    func=retrieve_insurance_info,
    description="Use this tool to answer insurance policy questions."
)

# ========= 🔹 ReAct Agent for Step-by-Step Reasoning mem =========
agent = initialize_agent(
    tools= [insurance_tool_forced_FAISS_retrival],#[insurance_tool],  # ✅ Uses the properly wrapped tool
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=20,
    max_execution_time=30,
    handle_parsing_errors=True,
    verbose=True
)

# ========= 🔹 API Endpoints 🔹 =========
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    template_question = """
    user question: {user_question}

    Search for the user question related info in context of both Zurich and Sompo products if the product or plan name is not mentioned.
    """
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = agent.run(template_question)  # ✅ Uses FAISS retrieval with enforced context
        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
@app.route("/recommend_mock", methods=["POST"])
def recommend_mock():
    category = request.form.get("category", "")  # Default to International    
    destination = request.form.get("destination", "")
    print(category, destination)
    template = ""  # Ensure template is always assigned
    travel_data ={}
    if category == "International":
        template =  """You are an AI travel insurance assistant. Your goal is to recommend the best travel insurance product **(Zurich or Sompo)** based on the latest travel insights.

                Latest Travel Insights:
                - {tavily_summary}

                Provide a recommendation for **someone visiting {destination}**, ensuring:
                1. **Select the best insurance provider** (Zurich or Sompo) based on coverage.
                2. **List the plan name** and key benefits.
                3. **Explain why this plan is best** based on risks and travel conditions.

                ### **Recommended Insurance Plan**
                - **Provider:** Zurich / Sompo
                - **Plan Name:** [Plan name]
                - **Key Benefits:** 
                - [Coverage 1]
                - [Coverage 2]
                - [Coverage 3]
                - **Why This Plan?** [Explain reasoning]

                ### **Additional Add-On Recommendations**
                - **[Add-on 1]**: [Why needed]
                - **[Add-on 2]**: [Why needed]

                Ensure the response is **well-organized, professional, and easy to understand.**"""
        #"""You are an AI travel insurance assistant. Based on the latest travel insights:
        # - {tavily_summary}
        # Provide a personalized insurance recommendation focusing on International travel products for someone visiting {destination}.
        # Highlight important add-ons based on current risks."""
        for travel in travel_data_intenational:
            travel_destination = travel["destination"]
            start_date = travel["startDate"]
            end_date = travel["endDate"]
            event = travel["event"]
            if travel_destination == request.form.get("destination", ""):
                travel_data["answer"] = event
    elif category == "Domestic":
        template = """You are an AI travel insurance assistant. Based on the latest travel insights:
        - {tavily_summary}
        Provide a personalized insurance recommendation focusing on Domestic travel products for someone visiting {destination}.
        Highlight important add-ons based on current risks."""
        for i in travel_data_domestic:
            travel_destination = travel["destination"]
            start_date = travel["startDate"]
            end_date = travel["endDate"]
            event = travel["event"]
            if travel_destination["destination"] == request.form.get("destination", ""):
                travel_data["answer"] = event
    if not template:
        return jsonify({"error": "Invalid category provided"}), 400

    if not travel_data or "answer" not in travel_data:
       return jsonify({"error": "Failed to fetch travel data"}), 500
    
    print("evendata", travel_data)

    prompt = PromptTemplate(input_variables=["destination", "tavily_summary"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    recommendation = chain.run(destination=destination, tavily_summary=travel_data["answer"])

    return jsonify({"recommendation": recommendation})



@app.route("/recommend", methods=["POST"])
def recommend():
    category = request.form.get("category", "International")  # Default to International    
    template = ""  # Ensure template is always assigned
    if category == "International":
        template = """You are an AI travel insurance assistant. Based on the latest travel insights:
        - {tavily_summary}
        Provide a personalized insurance recommendation focusing on International travel products for someone visiting {destination}.
        Highlight important add-ons based on current risks."""
    
    elif category == "Domestic":
        template = """You are an AI travel insurance assistant. Based on the latest travel insights:
        - {tavily_summary}
        Provide a personalized insurance recommendation focusing on Domestic travel products for someone visiting {destination}.
        Highlight important add-ons based on current risks."""
    
    if not template:
        return jsonify({"error": "Invalid category provided"}), 400

    destination = "France"
    travel_data = fetch_travel_data(destination, "March 1, 2025", "March 10, 2025")

    if not travel_data or "answer" not in travel_data:
        return jsonify({"error": "Failed to fetch travel data"}), 500

    prompt = PromptTemplate(input_variables=["destination", "tavily_summary"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    recommendation = chain.run(destination=destination, tavily_summary=travel_data["answer"])

    return jsonify({"recommendation": recommendation})

# ========= 🔹 Helper Functions 🔹 =========
def generate_recommendation(user_question, destination):
    # Get coverage info
    coverage_info = qa_chain.run(user_question)
    # Get live events for destination
    events =fetch_travel_data(destination)
    # Craft a combined prompt for the LLM:
    combined_prompt = (
        f"Based on the following policy details:\n{coverage_info}\n\n"
        f"and considering these recent events in {destination}:\n{', '.join(events)}\n\n"
        "What are the best recommendations for insurance coverage and add-ons? "
        "Answer concisely."
    )
    recommendation = llm(combined_prompt)
    return recommendation
    
def fetch_travel_data(destination, start_date, end_date):
    query = f"Upcoming events, holidays, health alerts, and travel risks in {destination} for duration {start_date} to {end_date} or in general."
    results = tavily.search(query=query, search_depth="advanced", include_answer=True)
    
    # Extract useful information
    return {
        "answer": results.get("answer", "No AI summary available"),
        "search_results": [{"title": r["title"], "url": r["url"]} for r in results.get("results", [])]
    }

def general_online_queries(query):
    results = results = tavily.search(query=query, search_depth="advanced", include_answer=True) or {} #tavily.search(query=query, search_depth="advanced", include_answer=True)
    return {
        "answer": results.get("answer", "No AI summary available"),
        "search_results": [
            {"title": r["title"], "url": r["url"]}
            for r in results.get("results", [])
        ] or [{"title": "No relevant results found", "url": ""}]
    }

# ========= 🔹 Run Flask App 🔹 =========
if __name__ == "__main__":
    app.run(debug=True)



# import os
# import requests
# import fitz  # PyMuPDF
# import re
# from flask import Flask, request, jsonify
# from langchain_openai import OpenAIEmbeddings, OpenAI
# # from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.text_splitter import CharacterTextSplitter
# # from langchain.chains import HuggingFaceHub
# from langchain_huggingface import HuggingFaceEndpoint
# from flask_cors import CORS
# from tavily import TavilyClient
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import MultiQueryRetriever

# app = Flask(__name__)
# CORS(app)

# # os.environ["OPENAI_API_KEY"] = "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"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# tavily = TavilyClient(api_key="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")
# policy_text = ""

# POLICY_DOCUMENT_PATH = ["sompodom_merged.pdf", "Zurich_APAC.pdf"] 
# print("log1")
# try:
#     policy_texts = []
#     for file_path in POLICY_DOCUMENT_PATH:
#         with fitz.open(file_path) as doc:
#             text = "\n".join([page.get_text() for page in doc])
#             policy_texts.append(text)
#             # print(f"Policy document ({file_path}):", text)

#     # Combine texts if needed
#     policy_text = "\n\n".join(policy_texts) 
#     print(policy_text)
# except FileNotFoundError:
#     print(f"Error: Policy document not found at {POLICY_DOCUMENT_PATH}")
# except Exception as e:
#     print(f"Error reading policy document: {e}")

# print("log2")
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)  # Adjust chunk size as needed
# texts = text_splitter.split_text(policy_text)


# print("log3")
# # embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.from_texts(texts, embeddings)

# print("log4")
# # llm = OpenAI(temperature=0)
# # llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B", model_kwargs={"temperature": 0.5,"max_length": 200, 
# #     "top_p": 0.3}) #meta-llama/Meta-Llama-3-8B

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-70B", 
#     temperature=0.2,  # Move outside model_kwargs
#     top_p=0.2,        # Move outside model_kwargs
#     task= "text-generation",  # Move outside model_kwargs
#     model_kwargs={"max_length": 2048}  # Keep only max_length inside
# )

# print("log5")
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm, retriever=db.as_retriever(), memory=memory, verbose=True
# )
# # qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(),verbose=False)    
# print("log6")
# # ========== API ROUTES ==========0

# # @app.route("/ask", methods=["POST"])
# # def ask():
# #     user_question = request.json.get("question")
# #     if not user_question:
# #         return jsonify({"error": "No question provided"}), 400
# #     # response = general_online_queries(user_question)
# #     # return jsonify(response)
# #     try:
# #         print("log7")
# #         print("Checking Hugging Face API key:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
# #         response = qa_chain.run(user_question)
# #         print("log8")
# #         pattern = r"Helpful Answer:(.*)"
# #         match = re.search(pattern, response, re.DOTALL)
# #         if match:
# #             helpful_answer = match.group(1).strip()
# #             print("Helpful Answer and Text After It:")
# #             print(helpful_answer)
# #             return jsonify({"answer": helpful_answer})
# #         else:
# #             print("No helpful answer found.")
            
# #     except Exception as e:
# #         import traceback
# #         print("Full error traceback:", traceback.format_exc())
# #         return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# @app.route("/ask", methods=["POST"])
# def ask():
#     user_question = request.json.get("question")
#     if not user_question:
#         return jsonify({"error": "No question provided"}), 400

#     try:
#         print("log7")
#         response = qa_chain.run(user_question)
#         return jsonify(response)
#         print("log8")
        
#         pattern = r"Helpful Answer:(.*)"
#         match = re.search(pattern, response, re.DOTALL)
        
#         if match:
#             helpful_answer = match.group(1).strip()
#             print("Helpful Answer and Text After It:")
#             print(helpful_answer)
#             return jsonify({"answer": helpful_answer})
#         else:
#             print("No helpful answer found.")
#             return jsonify({"answer": "Sorry, I couldn't find a relevant response."})  # Return fallback response

#     except Exception as e:
#         import traceback
#         print("Full error traceback:", traceback.format_exc())  # Print full error details
#         return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
    
# @app.route("/recommend/plan", methods=["POST"])
# def recommend():
#     template = """
#     You are an AI travel insurance assistant. Based on the latest travel insights:

#     - {tavily_summary}

#     Provide a personalized insurance recommendation for someone visiting {destination}.
#     Highlight important add-ons based on current risks.
#     """
#     travel_data = fetch_travel_data("France")
#     prompt = PromptTemplate(input_variables=["destination", "tavily_summary"], template=template)
#     chain = LLMChain(llm=llm, prompt=prompt)
#     recommendation = chain.run(destination="France", tavily_summary=travel_data["answer"])
#     print("AI Insurance Recommendation:\n", recommendation)


# # ========== HELPER FUNCTIONS ==========
# def generate_recommendation(user_question, destination):
#     # Get coverage info
#     coverage_info = qa_chain.run(user_question)
#     # Get live events for destination
#     events =fetch_travel_data(destination)
#     # Craft a combined prompt for the LLM:
#     combined_prompt = (
#         f"Based on the following policy details:\n{coverage_info}\n\n"
#         f"and considering these recent events in {destination}:\n{', '.join(events)}\n\n"
#         "What are the best recommendations for insurance coverage and add-ons? "
#         "Answer concisely."
#     )
#     recommendation = llm(combined_prompt)
#     return recommendation

# def fetch_travel_data(destination, start_date, end_date):
#     query = f"Upcoming events, holidays, health alerts, and travel risks in {destination} for duration {start_date} to {end_date} or in general."
#     results = tavily.search(query=query, search_depth="advanced", include_answer=True)
    
#     # Extract useful information
#     return {
#         "answer": results.get("answer", "No AI summary available"),
#         "search_results": [{"title": r["title"], "url": r["url"]} for r in results.get("results", [])]
#     }

# def general_online_queries(query):
#     results = results = tavily.search(query=query, search_depth="advanced", include_answer=True) or {} #tavily.search(query=query, search_depth="advanced", include_answer=True)
#     return {
#         "answer": results.get("answer", "No AI summary available"),
#         "search_results": [
#             {"title": r["title"], "url": r["url"]}
#             for r in results.get("results", [])
#         ] or [{"title": "No relevant results found", "url": ""}]
#     }

# if __name__ == "__main__" :
#     app.run(debug=True)


    

# #     # "am i covered for covid 19 disease in Zurich Domestic Travel Insurance, is it an add on benefit or pre included?"
# #     # i don't think covid 19 cover is pre-included in zurich travel domestic policy.
# #     # is covid 19 pre-included in zurich travel domestic policy.
# #     # how many plans does Zurich international travel have?
    

#     # rec = generate_recommendation(
#     # "Will I be covered in case of war and civil unrest for Zurich Travel Insurance? Yes or No.",
#     # "France"
#     # )
    
# # def fetch_travel_events(destination):
# #     API_KEY = "your_news_api_key"
# #     url = f"https://newsapi.org/v2/everything?q={destination}+travel+disaster&apiKey={API_KEY}"
# #     response = requests.get(url).json()
# #     articles = [article['title'] for article in response.get('articles', [])]
# #     return articles

#     # return {
#     #     "answer": results.get("answer", "No AI summary available"),
#     #     "search_results": [{"title": r["title"], "url": r["url"]} for r in results.get("results", [])]
#     # }