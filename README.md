# LLMprover

LLM-driven automatic theorem proving for Rocq/Coq, based on lemma-decomposition strategies.

## What works today

- Compile and validate `.v` scripts via `coqc` (statement follows attempt polarity: `P` or `~ (P)`)
- Multi-provider LLM client (OpenAI, Anthropic, Mistral)
- Domain types: `Goal`, `Polarity`, `ProofAttempt`, `AttemptRecord`, `LemmaNode`
- LLM prover agents: direct, repair, decomposition (`prove(node, polarity)`)
- Test suite with mocked LLM calls

## Design snapshot

- Each lemma is identified by a single `Goal` whose statement is a formula `P`.
- The search may try to prove `P` **or** its negation `~P`. Both kinds of attempts belong to the same lemma and are stored in two separate histories (positive and negative).
- If a proof of `~P` succeeds, the lemma is **refuted**: that outcome is an error for any parent that depended on proving `P`, and it must propagate upward.
- In the first version, `~P` is obtained by wrapping the statement as `~ (P)`.
- Agents offer three generation strategies — **direct**, **repair**, and **decomposition** — each usable on either polarity (`P` or `~P`).

## Roadmap

- [ ] Orchestration + strategies (walk `LemmaNode` trees with dual attempt histories)
- [ ] **`check_attempt` assumption check** — after a successful compile, verify that the main goal's proof depends only on lemmas listed in `new_lemmas` (e.g. via Rocq `Print Assumptions`)
- [ ] **Library imports** — `Require Import` / prelude configuration
- [ ] **Custom axiom environments** — goals over a user-provided context
- [ ] Interactive proving mode with Pytanque

## Quick start

```bash
pip install -e ".[dev]"
python3 -m pytest
python main.py   # DirectAgent + LemmaNode: prove, check, append, print status
```

## API keys

`main.py` loads variables from a `.env` file at the project root (via `python-dotenv`). Create one with the key for your provider:

```bash
# .env  (gitignored — do not commit)
MISTRAL_API_KEY=your-key-here
```

`main.py` currently uses Mistral (`MistralAIClient.from_env()`). The LLM client module also supports OpenAI and Anthropic via the same pattern:

| Provider  | Environment variable   |
|-----------|------------------------|
| Mistral   | `MISTRAL_API_KEY`      |
| OpenAI    | `OPENAI_API_KEY`       |
| Anthropic | `ANTHROPIC_API_KEY`    |

Alternatively, export the variable in your shell before running:

```bash
export MISTRAL_API_KEY=your-key-here
python main.py
```

Shell exports take precedence over `.env` if both are set.

## Architecture

```
main.py
   │
   ├── domain.py         Goal, Polarity, ProofAttempt, CoqcResult, AttemptRecord, LemmaNode
   ├── proof_script.py   ProofScript (.v file I/O)
   ├── rocq.py           CoqcBackend — assemble scripts, run coqc
   ├── llm_client.py     LLMClient — OpenAI, Anthropic, Mistral
   ├── prompts.py        Prompt load / fill
   └── prover_agents/    Direct, repair, decomposition (+ utils)
```

**Dependency flow** (each layer only imports from layers above it):

```
proof_script  →  domain  →  rocq
                         →  llm_client
                         →  prover_agents
```

| Module | Role |
|--------|------|
| `proof_script.py` | Rocq source text (`ProofScript`, `from_file`) |
| `domain.py` | `Goal`, `Polarity`, `ProofAttempt`, `CoqcResult`, `AttemptRecord`, `LemmaNode` (dual attempt histories + status) |
| `rocq.py` | Runs `coqc` on a `ProofScript` or a full `ProofAttempt` (main statement from polarity) |
| `llm_client.py` | Stateless chat-completion wrapper over LLM provider SDKs |
| `prover_agents/` | `ProverAgent.prove(node, polarity)` implementations and shared helpers |

**Planned:** `Orchestrator`, `OrchestrationStrategy`, `AgentRegistry`, `pytanque_session`.

### Design notes

**`ProofScript`** is a thin dataclass around a `str` (Rocq source). A plain string would work for most of the current code; the class is kept as a hook for script-level operations (`from_file`, future `replace_lemma`, etc.).

**`LemmaNode`** is the search unit for one lemma: canonical `Goal` (`P`), positive and negative attempt lists, and a derived status (`Open` / `Proved` / `Refuted`). **`AttemptRecord`** stores a checked `ProofAttempt`, the `CoqcResult`, and child lemma attempt lists for decomposition.

**Polarity.** For a lemma with statement `P`, an attempt is either positive (try to prove `P`) or negative (try to prove `~P`). The lemma’s identity stays `P`; each `ProofAttempt` records which polarity was attacked. When the negated statement is needed (prompts, `coqc` scripts), the first version builds it as `~ (P)`.

**Decomposition invariant (not enforced yet).** When `new_lemmas` is non-empty, a valid parent proof should use *only* those admitted lemmas as logical dependencies. Today `check_attempt` only checks that the assembled script compiles.

**Current limitations.** Scripts are self-contained fragments: no `Require Import`, no configurable prelude, and no rich ambient theory. The demo in `main.py` uses bare `Prop`/`/\` tactics without imports.

### Proof loop (target design)

1. `ProverAgent.prove(node, polarity)` proposes a `ProofAttempt` for the lemma.
2. `CoqcBackend.make_script` assembles a complete `.v` file (using `P` or `~ (P)`) and `check_attempt` validates it.
3. On failure, the `CoqcResult` is stored in an `AttemptRecord` on that polarity’s history for repair or mode/polarity switch.
4. On negation success, mark the lemma refuted and propagate failure upward.
5. The orchestrator maintains lemma nodes and walks the search (details TBD).
