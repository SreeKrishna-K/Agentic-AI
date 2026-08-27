from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from utils.logging import log_panel, log_rule
from ingestion import retriever


def test_retrival_grader_answer_yes() -> None:
    question = "What is Long-term memory?"
    log_rule("TEST  ·  retrieval grader  ·  expect yes", "bold green")
    log_panel("Question", question, "cyan")

    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    log_panel(f"Retrieved document  ·  index 1 / {len(docs)}", doc_txt, "blue")

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    log_panel("Grade", res.binary_score, "green")

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    off_topic = "how to make pizaa"
    log_rule("TEST  ·  retrieval grader  ·  expect no", "bold red")
    log_panel("Retrieve query", question, "cyan")
    log_panel("Grade question (off-topic)", off_topic, "yellow")

    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    log_panel(f"Retrieved document  ·  index 1 / {len(docs)}", doc_txt, "blue")

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": off_topic, "document": doc_txt}
    )
    log_panel("Grade", res.binary_score, "red")

    assert res.binary_score == "no"
