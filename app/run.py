from flask import Flask, request, render_template, jsonify

from modules.rag_pipeline import RAGPipeline
from modules.generator import OllamaGenerator
from modules.retriever import QdrantRetriever
from modules.embedder import HuggingFaceEmbedder

from configs import FlaskConfig
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
app.config.from_object(FlaskConfig)

embedder = HuggingFaceEmbedder()
retriever = QdrantRetriever(embedder=embedder, base_url="host.docker.internal:6333")
llm_generator = OllamaGenerator(base_url="host.docker.internal:11434")

rag_pipeline = RAGPipeline(
    retriever=retriever, 
    llm_generator=llm_generator,
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST", "GET"])
def query_rag_pipeline():
    data = request.get_json()
    query = data.get("query")
    try:
        response, _ = rag_pipeline.query(query)
        return str(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ingest', methods=["POST"])
def ingest_documents():
    data = request.get_json()
    documents = data['documents']
    try:
        retriever.add(documents)
        return jsonify({"message": "Documents successfully ingested."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)