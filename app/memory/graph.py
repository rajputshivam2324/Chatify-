import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.services.llm_service import LLMInvocationError, invoke_llm

from . import long_term, short_term


class MessagesState(TypedDict):
    user_id: str
    session_id: str
    model_key: str
    messages: list[dict[str, Any]]
    long_term_context: str
    reply: str


async def retrieve_short_term(state: MessagesState) -> dict[str, Any]:
    """Retrieve short-term memory from Redis."""
    session_id = state["session_id"]
    messages = await short_term.get_messages(session_id)
    return {"messages": messages}


async def retrieve_long_term(state: MessagesState) -> dict[str, Any]:
    """Retrieve long-term memory from FAISS."""
    user_id = uuid.UUID(state["user_id"])
    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    context = await long_term.long_term_memory.search(user_id, user_message, k=5)
    return {"long_term_context": context}


async def call_llm(state: MessagesState) -> dict[str, Any]:
    """Call the LLM with full context."""
    model_key = state["model_key"]
    long_term_context = state["long_term_context"]

    system_prompt = "You are a helpful AI assistant."
    if long_term_context:
        system_prompt += f"\n\nRelevant past conversation context:\n{long_term_context}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(state["messages"])

    try:
        reply = await invoke_llm(model_key, messages)
    except LLMInvocationError as e:
        reply = f"I apologize, but I encountered an error processing your request. Please try again."

    return {"reply": reply}


async def save_short_term(state: MessagesState) -> dict[str, Any]:
    """Save to short-term memory in Redis."""
    session_id = state["session_id"]
    messages = state["messages"]

    if messages:
        user_msg = messages[-1]["content"]
        assistant_reply = state["reply"]
        await short_term.append_messages(session_id, user_msg, assistant_reply)

    return {}


async def save_long_term(state: MessagesState) -> dict[str, Any]:
    """Save to long-term memory in FAISS."""
    user_id = uuid.UUID(state["user_id"])
    messages = state["messages"]

    if messages:
        user_msg = messages[-1]["content"]
        assistant_reply = state["reply"]
        combined = f"User: {user_msg}\nAssistant: {assistant_reply}"
        await long_term.long_term_memory.add_memory(user_id, combined)

    return {}


def create_graph() -> StateGraph:
    """Create and compile the LangGraph."""
    builder = StateGraph(MessagesState)

    builder.add_node("retrieve_short_term", retrieve_short_term)
    builder.add_node("retrieve_long_term", retrieve_long_term)
    builder.add_node("call_llm", call_llm)
    builder.add_node("save_short_term", save_short_term)
    builder.add_node("save_long_term", save_long_term)

    builder.set_entry_point("retrieve_short_term")
    builder.add_edge("retrieve_short_term", "retrieve_long_term")
    builder.add_edge("retrieve_long_term", "call_llm")
    builder.add_edge("call_llm", "save_short_term")
    builder.add_edge("save_short_term", "save_long_term")
    builder.add_edge("save_long_term", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


graph = create_graph()


async def run_graph(
    user_id: uuid.UUID,
    session_id: str,
    model_key: str,
    user_message: str,
) -> str:
    """Run the memory graph and return assistant reply."""
    initial_state: MessagesState = {
        "user_id": str(user_id),
        "session_id": session_id,
        "model_key": model_key,
        "messages": [{"role": "user", "content": user_message}],
        "long_term_context": "",
        "reply": "",
    }

    config = {"configurable": {"thread_id": session_id}}

    result = await graph.ainvoke(initial_state, config)
    return result["reply"]