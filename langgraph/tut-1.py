from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")

# 1. Initialize the LLM
model = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY)

# 2. Define the node function
def chatbot_node(state: MessagesState):
    # The state automatically has a "messages" key containing the list
    response = model.invoke(state["messages"])
    # Return a dictionary updating the "messages" key
    return {"messages": [response]}

# 3. Build the graph using StateGraph
workflow = StateGraph(MessagesState)

# 4. Add the node and define the flow
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# 5. Compile the graph
app = workflow.compile()

# 6. Run the graph
input_messages = {"messages": [("user", "Hi! What is LangGraph?")]}
for event in app.stream(input_messages):
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)
