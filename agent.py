from typing import TypedDict, List
import ast

def _extract_text_from_content(content) -> str:
    """Extract plain text from Gemini response content.
    Handles strings, lists of dicts (e.g., [{'type': 'text', 'text': '...'}]),
    stringified Python literals, and unexpected structures safely.
    """
    if not content:
        return ""
    # Direct string case
    if isinstance(content, str):
        stripped = content.strip()
        # If the string looks like a list/dict representation, try to eval it.
        if stripped.startswith("[") and ("'text':" in stripped or '"text":' in stripped):
            try:
                parsed = ast.literal_eval(stripped)
                return _extract_text_from_content(parsed)
            except Exception:
                pass
        return stripped
    # List case – concatenate any 'text' fields.
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text", "")
                if isinstance(txt, str):
                    parts.append(txt)
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "".join(parts).strip()
    # Dict case – try to get a 'text' key.
    if isinstance(content, dict):
        txt = content.get("text", "")
        if isinstance(txt, str):
            return txt.strip()
        return str(content).strip()
    # Fallback – stringify.
    return str(content).strip()

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class AgentState(TypedDict):
    question: str
    context: str
    sources: List[str]
    answer: str


def create_agent(vector_store):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )

    # -------------------------
    # Retrieve relevant PDF text
    # -------------------------
    def retrieve(state: AgentState):

        question = state["question"]

        results = vector_store.search(
            question,
            k=4
        )

        context_parts = []
        sources = []

        for result in results:

            context_parts.append(
                result["text"]
            )

            source = f"Page {result['page']}"

            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

        return {
            "context": context,
            "sources": sources
        }

    # -------------------------
    # Generate answer
    # -------------------------
    def generate_answer(state: AgentState):

        question = state["question"]
        context = state["context"]

        if not context.strip():
            return {
                "answer": "I could not find this information in the uploaded PDF."
            }

        prompt = f"""
You are a helpful PDF Question Answering Agent.

Answer the user's question using ONLY the information
provided in the PDF context below.

Do not use outside knowledge.

Do not invent information.

If the answer cannot be found in the PDF context, say:

"I could not find this information in the uploaded PDF."

PDF CONTEXT:
-------------------------
{context}
-------------------------

USER QUESTION:
{question}

Give a clear, concise and easy-to-read answer.

Use bullet points when listing multiple items.
Do not include technical metadata, signatures, JSON, or internal information.
"""

        response = llm.invoke(prompt)

        # --------------------------------
        # Extract ONLY text from response
        # --------------------------------
        # Use robust extraction to get only the answer text.
        try:
            answer = _extract_text_from_content(response.content)
        except Exception as e:
            # Fallback: log error and return a generic message.
            answer = ""
        return {
            "answer": answer or "I could not find this information in the uploaded PDF."
        }

    # -------------------------
    # Create LangGraph workflow
    # -------------------------
    workflow = StateGraph(AgentState)

    workflow.add_node(
        "retrieve",
        retrieve
    )

    workflow.add_node(
        "generate",
        generate_answer
    )

    workflow.add_edge(
        START,
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()