import os
import pandas as pd
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from google import genai
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# 1. Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

if not all([GEMINI_API_KEY, SLACK_BOT_TOKEN, SLACK_APP_TOKEN]):
    raise ValueError("Missing GEMINI_API_KEY, SLACK_BOT_TOKEN, or SLACK_APP_TOKEN in .env file.")

# 2. Initialize Gemini and Slack Bolt
genai_client = genai.Client(api_key=GEMINI_API_KEY)
slack_app = App(token=SLACK_BOT_TOKEN)

AVAILABLE_GENERATION_MODEL = "gemini-3.5-flash-lite"

def get_embedding_model() -> str:
    try:
        models_list = list(genai_client.models.list())
        for model in models_list:
            actions = getattr(model, "supported_generation_methods", []) or getattr(model, "supported_actions", [])
            model_name = getattr(model, "name", "").replace("models/", "")
            if "embedContent" in actions or "embed_content" in actions:
                return model_name
    except Exception:
        pass
    return "text-embedding-004"

AVAILABLE_EMBEDDING_MODEL = get_embedding_model()

# 3. Embedding Function for ChromaDB
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            res = genai_client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            if hasattr(res, "embeddings") and res.embeddings:
                embeddings.append(res.embeddings[0].values)
            elif hasattr(res, "embedding") and res.embedding:
                embeddings.append(res.embedding.values)
            else:
                raise ValueError(f"Unexpected embedding output format: {res}")
        return embeddings

# 4. Vector Database Setup
def init_vector_db():
    if not os.path.exists("company_policies.csv"):
        raise FileNotFoundError("File 'company_policies.csv' not found.")

    df = pd.read_csv("company_policies.csv")
    chroma_client = chromadb.Client()
    gemini_emb_fn = GeminiEmbeddingFunction(model_name=AVAILABLE_EMBEDDING_MODEL)
    
    try:
        chroma_client.delete_collection("slack_policies_gemini")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="slack_policies_gemini", 
        embedding_function=gemini_emb_fn
    )
    
    documents, metadatas, ids = [], [], []
    for idx, row in df.iterrows():
        title = str(row.get('title', ''))
        dept = str(row.get('department', ''))
        cat = str(row.get('category', ''))
        policy = str(row.get('policy_text', ''))
        
        doc_text = f"Title: {title}\nDepartment: {dept}\nCategory: {cat}\nPolicy: {policy}"
        documents.append(doc_text)
        metadatas.append({"title": title, "department": dept, "category": cat})
        ids.append(f"policy_{idx}")
        
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection

print("Loading vector database...")
collection = init_vector_db()
print("Vector database initialized successfully!")

# 5. Answer Question using RAG
def answer_policy_question(user_query: str, top_k: int = 3) -> str:
    results = collection.query(query_texts=[user_query], n_results=top_k)
    retrieved_docs = results['documents'][0] if results and 'documents' in results else []
    context = "\n\n---\n\n".join(retrieved_docs)
    
    prompt = f"""You are an internal company policy assistant on Slack.
Answer the user's question EXCLUSIVELY using the context provided below.
Keep your response concise, professional, and clear for Slack formatting.
Identify and cite the relevant policy title in your answer.
If the information is not contained in the context, state clearly: "I do not have information regarding this policy in the official database."

Relevant Context:
{context}

Question: {user_query}
"""
    response = genai_client.models.generate_content(
        model=AVAILABLE_GENERATION_MODEL,
        contents=prompt
    )
    return response.text

# 6. Event Handlers
@slack_app.event("app_mention")
def handle_app_mention_events(body, say):
    event = body.get("event", {})
    user_query = event.get("text", "")
    if ">" in user_query:
        user_query = user_query.split(">", 1)[1].strip()
        
    say("🔎 *Searching company policies...*")
    answer = answer_policy_question(user_query)
    say(answer)

@slack_app.event("message")
def handle_message_events(body, say):
    event = body.get("event", {})
    if event.get("channel_type") == "im" and not event.get("bot_id"):
        user_query = event.get("text", "")
        say("🔎 *Searching company policies...*")
        answer = answer_policy_question(user_query)
        say(answer)

# 7. Start App
if __name__ == "__main__":
    print("⚡ Slack Policy Bot is running!")
    handler = SocketModeHandler(slack_app, SLACK_APP_TOKEN)
    handler.start()