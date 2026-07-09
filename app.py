import gradio as gr

from rag import chat


def respond(message, history):
    return chat(message)


demo = gr.ChatInterface(
    fn=respond,
    title="RAG Chatbot with OpenAI & Pinecone",
    description="Ask questions about the indexed SQuAD dataset.",
)

demo.launch()