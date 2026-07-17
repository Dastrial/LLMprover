"""CLI entry point."""

from dotenv import load_dotenv

from llmprover.domain import AttemptRecord, Goal, LemmaNode, Polarity
from llmprover.llm_client import MistralAIClient
from llmprover.prover_agents.direct_agent import DirectAgent
from llmprover.rocq import CoqcBackend

load_dotenv()  # .env: MISTRAL_API_KEY=...  (or: export MISTRAL_API_KEY=...)

if __name__ == "__main__":
    coqc = CoqcBackend()
    goal = Goal(statement="forall A B : Prop, A -> B -> A /\\ B", name="and_statement")
    node = LemmaNode(goal=goal)
    agent = DirectAgent(MistralAIClient.from_env())

    attempt = agent.prove(node, Polarity.Positive)
    print(attempt.script)

    result = coqc.check_attempt(attempt)
    node.append(AttemptRecord(attempt=attempt, rocq_error=result))
    print(result)
    print(node.status)
