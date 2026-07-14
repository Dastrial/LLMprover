"""CLI entry point."""

from dotenv import load_dotenv

from llmprover.domain import Goal, ProofAttempt
from llmprover.llm_client import MistralAIClient
from llmprover.rocq import CoqcBackend

load_dotenv()  # .env: MISTRAL_API_KEY=...  (or: export MISTRAL_API_KEY=...)

if __name__ == "__main__":
    coqc = CoqcBackend()
    goal = Goal(statement="forall A B : Prop, A -> B -> A /\\ B", name="and_statement")
    llm_client = MistralAIClient.from_env()
    prompt = (
        f"Write the Rocq proof body for: {goal.statement}. "
        "Output only tactics between Proof. and Qed. "
        "(no ```coq, ```rocq, Proof./Qed., comments, explanations, markdown fences, or extra text)."
    )
    llm_answer = llm_client.complete([{"role": "user", "content": prompt}])
    print(llm_answer)
    llm_attempt = ProofAttempt(goal=goal, script=llm_answer, new_lemmas=[])
    result = coqc.check_attempt(llm_attempt)
    print(result)
