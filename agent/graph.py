from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv;
load_dotenv()
from .prompts import *
from .states import *
from langgraph.constants import END 
from langgraph. graph import StateGraph
from langgraph.prebuilt import create_react_agent
import os


if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')

# model = ChatGroq(model="openai/gpt-oss-120b")
model = ChatGroq(model="openai/gpt-oss-20b")

from .tools import *

from langchain.globals import set_verbose, set_debug
# set_debug (True)
# set_verbose (True)



def planner_agent(state: dict) -> dict:
    user_prompt = state["user_prompt"]
    resp = model.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    if resp is None:
        raise ValueError("Planner did not return a valid response.")
    return { "plan": resp }

def architect_agent(state: dict)-> dict:
    plan: Plan = state['plan']
    resp = model.with_structured_output(TaskPlan).invoke(architect_prompt(plan))
    resp.plan = plan
    if resp is None: 
        raise ValueError("Architect did not return a valid response.")
    return { "task_plan": resp }

def coder_agent(state: dict)-> dict:
    coder_state: TaskPlan = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"])

    steps = coder_state.task_plan.implementation_steps

    if coder_state.current_step_idx >= len(steps) :
        return {"coder_state": coder_state, "status": "DONE"}
    
    current_task = steps[coder_state.current_step_idx]

    existing_content= read_file.run(current_task.filepath)
    user_prompt = (f"Task: {current_task.task_description}\n"
                   f"File: {current_task.filepath}\n"
                    f"Existing content:\n{existing_content}\n"
                    "Use write_file(path, content) to save your changes.")
    
    system_prompt = coder_system_prompt() 
    coder_tools = [read_file, write_file, list_files, get_current_directory]
    react_agent = create_react_agent(model, coder_tools)
    react_agent.invoke({"messages":[{"role": "system", "content": system_prompt},
                                    {"role": "user"  , "content": user_prompt}]})
    
    coder_state.current_step_idx += 1

    return {"coder_state": coder_state}


graph = StateGraph(dict)
graph.add_node ("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge("planner","architect")
graph.add_edge("architect", "coder")
graph.add_conditional_edges("coder", 
                            lambda s: "END" if s.get("status") == "DONE" else "coder",
                            {"END": END, "coder": "coder"})
graph.set_entry_point("planner")
agent = graph.compile()

# user_prompt = "Create a to-do list application using html, css, and javascript."
# agent.invoke({"user_prompt": user_prompt},
#             {"recursion_limit": 100})