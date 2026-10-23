from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY


embeddings = download_embeddings()

index_name = "medical-rag"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

chatModel = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    streaming=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()

    if not msg:
        return jsonify({"error": "Please enter a question."}), 400

    @stream_with_context
    def generate():
        try:
            # Retrieve relevant documents using the original retriever.
            docs = retriever.invoke(msg)

            # Keep the original RAG prompt and context format.
            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            # Stream the answer directly from the original chat model.
            messages = prompt.format_messages(
                input=msg,
                context=context
            )

            for chunk in chatModel.stream(messages):
                content = chunk.content

                if isinstance(content, str):
                    if content:
                        yield content

                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text", "")
                            if text:
                                yield text
                        elif isinstance(item, str):
                            yield item

        except Exception:
            app.logger.exception("Error while streaming the RAG response")
            yield "\n\nSorry, something went wrong. Please try again."

    return Response(
        generate(),
        content_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True,
        threaded=True
    )
