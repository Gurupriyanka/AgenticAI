from typing import TypedDict, List, Dict
import logging
from datetime import datetime
import os

from tavily import TavilyClient
from pydantic import BaseModel

from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run, Resource
from lib.llm import LLM
from lib.messages import BaseMessage, UserMessage, SystemMessage, AIMessage
from lib.vector_db import VectorStore



logging.getLogger("pdfminer").setLevel(logging.ERROR)


class EvaluationReport(BaseModel):
    useful: bool
    description: str


class AgenticRAGState(TypedDict, total=False):
    """
    Type definition for the state object passed through the Agentic RAG pipeline.
    """
    question: str
    messages: List[BaseMessage]
    documents: List[str]
    metadatas: List[dict]
    distances: List[float]
    evaluation_useful: bool
    evaluation_description: str
    web_results: Dict
    answer_source: str
    answer: str


class AgenticRAG:
    """
    Agentic Retrieval-Augmented Generation (Agentic RAG) system.

    This class extends a basic RAG pipeline by adding:
    1. Retrieval from the internal vector database
    2. Evaluation of retrieval quality
    3. Web search fallback when retrieval is insufficient
    4. Final answer generation using the best available context
    """

    def __init__(self, llm: LLM, vector_store: VectorStore, web_search_tool):
        #self.workflow = self._create_state_machine()
        self.resource = Resource(
            vars={
                "llm": llm,
                "vector_store": vector_store,
                "web_search_tool": web_search_tool,
            }
        )
        self.workflow = self._create_state_machine()

    def _retrieve(self, state: AgenticRAGState, resource: Resource) -> AgenticRAGState:
        vector_store = resource.vars.get("vector_store")
        question = state["question"]

        results = vector_store.query(
            query_texts=[question],
            n_results=3
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        #print("RETRIEVED METADATAS:", metadatas)

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }


    import json
    import ast

    def _evaluate_retrieval(self, state: AgenticRAGState, resource: Resource) -> AgenticRAGState:
        llm: LLM = resource.vars.get("llm")
        question = state["question"]
        metadatas = state.get("metadatas", [])

        messages = [
            UserMessage(
                content=(
                    "You are evaluating whether retrieved video game database records are sufficient "
                    "to answer a user's question.\n\n"
                    f"User question: {question}\n\n"
                    f"Retrieved metadata records: {metadatas}\n\n"
                    "Return an object with keys 'useful' and 'description'. "
                    "Mark useful=true if the retrieved records contain enough information to answer "
                    "the question directly or with a simple inference from fields such as "
                    "Name, Platform, Publisher, Genre, YearOfRelease, or Description. "
                    "Mark useful=false only if the retrieved records are missing the needed facts "
                    "or are clearly unrelated."
                )
            )
        ]

        judge_response = llm.invoke(messages, response_format=EvaluationReport)
        report = judge_response.content

        # print("RAW REPORT:", report)
        # print("REPORT TYPE:", type(report))

        useful = False
        description = "Unexpected evaluation format."

        if hasattr(report, "useful") and hasattr(report, "description"):
            useful = bool(report.useful)
            description = str(report.description)

        elif isinstance(report, dict):
            useful = bool(report.get("useful", False))
            description = str(report.get("description", "No description provided."))

        elif isinstance(report, str):
            try:
                parsed = json.loads(report)
                useful = bool(parsed.get("useful", False))
                description = str(parsed.get("description", "No description provided."))
            except Exception:
                try:
                    parsed = ast.literal_eval(report)
                    useful = bool(parsed.get("useful", False))
                    description = str(parsed.get("description", "No description provided."))
                except Exception:
                    if '"useful":true' in report.lower():
                        useful = True
                    elif '"useful":false' in report.lower():
                        useful = False
                    description = report

        # print("NORMALIZED USEFUL:", useful)
        # print("NORMALIZED DESCRIPTION:", description)

        return {
            "evaluation_useful": useful,
            "evaluation_description": description,
        }


    def _web_search(self, state: AgenticRAGState, resource: Resource) -> AgenticRAGState:
        print("WEB SEARCH CHECK - evaluation_useful:", state.get("evaluation_useful"))

        if state.get("evaluation_useful", False):
            return {
                "web_results": {},
                "answer_source": "retrieval",
            }

        question = state["question"]
        web_search_tool = resource.vars.get("web_search_tool")
        web_results = web_search_tool(question)

        return {
            "web_results": web_results,
            "answer_source": "web",
        }




    def _augment(self, state: AgenticRAGState) -> AgenticRAGState:
        question = state["question"]

        if state.get("evaluation_useful", False):
            context = str(state.get("metadatas", []))
            source = "retrieval"
        else:
            web_results = state.get("web_results", {})
            answer = web_results.get("answer", "")
            results = web_results.get("results", [])
            context = f"{answer}\n\n{results}"
            source = "web"

        messages = [
            SystemMessage(
                content=(
                    "You are a game industry research assistant. "
                    "Answer the user's question using only the provided context. "
                    "If the context is insufficient, say that you do not know."
                )
            ),
            UserMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Source: {source}\n\n"
                    f"Context:\n{context}\n\n"
                    "Answer:"
                )
            )
        ]

        return {
            "messages": messages,
            "answer_source": source
        }

    def _generate(self, state: AgenticRAGState, resource: Resource) -> AgenticRAGState:
        llm: LLM = resource.vars.get("llm")
        ai_message = llm.invoke(state["messages"])

        return {
            "answer": ai_message.content,
            "messages": state["messages"] + [ai_message],
        }

    def _create_state_machine(self) -> StateMachine[AgenticRAGState]:
        machine = StateMachine[AgenticRAGState](AgenticRAGState)

        entry = EntryPoint[AgenticRAGState]()
        retrieve = Step[AgenticRAGState]("retrieve", self._retrieve)
        evaluate = Step[AgenticRAGState]("evaluate_retrieval", self._evaluate_retrieval)
        web_search = Step[AgenticRAGState]("web_search", self._web_search)
        augment = Step[AgenticRAGState]("augment", self._augment)
        generate = Step[AgenticRAGState]("generate", self._generate)
        termination = Termination[AgenticRAGState]()

        machine.add_steps([entry, retrieve, evaluate, web_search, augment, generate, termination])
        machine.connect(entry, retrieve)
        machine.connect(retrieve, evaluate)
        machine.connect(evaluate, web_search)
        machine.connect(web_search, augment)
        machine.connect(augment, generate)
        machine.connect(generate, termination)

        return machine

    def invoke(self, query: str) -> Run:
        initial_state: AgenticRAGState = {
            "question": query,
        }

        run_object = self.workflow.run(
            state=initial_state,
            resource=self.resource,
        )
        return run_object


