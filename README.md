# LLMprover

LLM-driven automatic theorem proving for Rocq/Coq, based on lemma-decomposition strategies and dual-polarity search (prove `P` or refute it by proving `~P`).

## What works today

- **Orchestrator** — search loop over a `LemmaNode` tree with token/attempt budgets
- **Lemma selection** — `SimpleLemmaSelectionStrategy` (least-attempted open nodes, polarity alternation, frontier/closure propagation)
- **Attempt strategy** — `LLMAttemptStrategy` picks agent + model via an LLM stratège
- **Registries** — `AgentRegistry` and `ModelRegistry` (lazy instantiation, catalogue specs for the stratège)
- **Prover agents** — direct, repair-direct, decomposition (`prove(node, polarity)`)
- **Rocq checking** — assemble and validate `.v` scripts via `coqc` (statement follows attempt polarity: `P` or `~ (P)`)
- **Multi-provider LLM client** — OpenAI, Anthropic, Mistral (with `TokenUsage` tracking)
- **Domain types** — `Goal`, `Polarity`, `Position`, `ProofAttempt`, `AttemptRecord`, `LemmaNode`
- Test suite with mocked LLM / `coqc` calls

## Design snapshot

- Each lemma is identified by a single `Goal` whose statement is a formula `P`.
- The search may try to prove `P` **or** its negation `~P`. Both kinds of attempts belong to the same lemma and are stored in two separate histories (positive and negative).
- If a proof of `~P` succeeds, the lemma is **refuted**: that outcome is an error for the parent that depended on proving `P`, and it must propagate upward.
- In the first version, `~P` is obtained by wrapping the statement as `~ (P)`.
- Agents offer three generation strategies — **direct**, **repair**, and **decomposition** — each usable on either polarity (`P` or `~P`).
- The orchestrator repeatedly: select (position, polarity) → decide agent+model → generate attempt → `coqc` check → update histories and selection state.

## Roadmap

- [x] Orchestration + strategies (walk `LemmaNode` trees with dual attempt histories)
- [ ] **`check_attempt` assumption check** — after a successful compile, verify that the main goal's proof depends only on lemmas listed in `new_lemmas` (e.g. via Rocq `Print Assumptions`)
- [ ] **Library imports** — `Require Import` / prelude configuration
- [ ] **Custom axiom environments** — goals over a user-provided context
- [ ] Interactive proving mode with Pytanque (`pytanque_session.py` is a stub)

## Quick start

```bash
pip install -e ".[dev]"
python3 -m pytest
python main.py   # Orchestrator on and_or_distrib (needs MISTRAL_API_KEY + coqc)
```

## API keys

`main.py` loads variables from a `.env` file at the project root (via `python-dotenv`). Create one with the key for your provider:

```bash
# .env  (gitignored — do not commit)
MISTRAL_API_KEY=your-key-here
```

`main.py` currently wires Mistral models (`mistral-small-latest` as stratège + catalogue; `mistral-large-latest` as stronger option). The LLM client module also supports OpenAI and Anthropic via the same pattern:

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
   ├── orchestrator.py              Orchestrator — search loop
   ├── attempt_strategy.py          AttemptStrategy (ABC)
   ├── llm_attempt_strategy.py      LLM picks agent + model
   ├── lemma_selection_strategy.py  LemmaSelectionStrategy (ABC)
   ├── simple_lemma_selection_strategy.py
   ├── agent_registry.py            Agent catalogue
   ├── model_registry.py            Model catalogue
   ├── domain.py                    Goal, Polarity, Position, ProofAttempt, …
   ├── proof_script.py              ProofScript (.v file I/O)
   ├── rocq.py                      CoqcBackend — assemble scripts, run coqc
   ├── llm_client.py                LLMClient — OpenAI, Anthropic, Mistral
   ├── prompts.py                   Prompt load / fill
   ├── prompts/                     Strategy templates
   ├── utils.py                     Parse LLM output, format histories
   ├── pytanque_session.py          Stub (future interactive mode)
   └── prover_agents/               Direct, repair, decomposition + prompts
```

**Dependency flow** (simplified):

```
proof_script  →  domain  →  rocq
                         →  llm_client
                         →  utils / prompts  →  prover_agents
                                            →  registries
                                            →  strategies  →  orchestrator
```

| Module | Role |
|--------|------|
| `domain.py` | `Goal`, `Polarity`, `Position`, `ProofAttempt`, `CoqcResult`, `AttemptRecord`, `LemmaNode` |
| `rocq.py` | Runs `coqc` on a `ProofScript` or a full `ProofAttempt` (main statement from polarity) |
| `llm_client.py` | Stateless chat-completion wrapper; returns `CompletionResult` + `TokenUsage` |
| `agent_registry.py` / `model_registry.py` | Catalogues with specs for the stratège; lazy `get` |
| `LLMAttemptStrategy` | Chooses `(ProverAgent, TokenUsage)` from histories + catalogues |
| `SimpleLemmaSelectionStrategy` | Chooses `(Position, Polarity)`; manages open set and closure |
| `Orchestrator` | Budgeted loop: select → decide → prove → check → append → update |
| `prover_agents/` | `ProverAgent.prove(node, polarity)` implementations |

### Design notes

**`ProofScript`** is a thin dataclass around a `str` (Rocq source). A plain string would work for most of the current code; the class is kept as a hook for script-level operations (`from_file`, future `replace_lemma`, etc.).

**`LemmaNode`** is the search unit for one lemma: canonical `Goal` (`P`), positive and negative attempt lists, and a derived status (`Open` / `Proved` / `Refuted`). **`AttemptRecord`** stores a checked `ProofAttempt`, the `CoqcResult`, and child lemma attempt lists for decomposition. Navigation in the tree uses **`Position`** (`from_position`).

**Polarity.** For a lemma with statement `P`, an attempt is either positive (try to prove `P`) or negative (try to prove `~P`). The lemma’s identity stays `P`; each `ProofAttempt` records which polarity was attacked. When the negated statement is needed (prompts, `coqc` scripts), the first version builds it as `~ (P)`.

**Decomposition invariant (not enforced yet).** When `new_lemmas` is non-empty, a valid parent proof should use *only* those admitted lemmas as logical dependencies. Today `check_attempt` only checks that the assembled script compiles.

**Current limitations.** Scripts are self-contained fragments: no `Require Import`, no configurable prelude, and no rich ambient theory. The demo in `main.py` uses bare `Prop`/`/\`/`\/`/`<->` without imports.

### Proof loop (implemented)

1. `LemmaSelectionStrategy.select_lemma()` picks a `(Position, Polarity)`.
2. `AttemptStrategy.decide_agent(node, polarity)` picks a `ProverAgent` (and underlying model).
3. `ProverAgent.prove(node, polarity)` proposes a `ProofAttempt`.
4. `CoqcBackend.make_script` assembles a complete `.v` file (using `P` or `~ (P)`) and `check_attempt` validates it.
5. An `AttemptRecord` is appended; child `LemmaNode`s are created for `new_lemmas` (unique names).
6. `LemmaSelectionStrategy.update` opens children on coqc success, forgets superseded frontiers, and propagates closure (`Proved` / `Refuted`).
7. On negation success, the lemma is refuted; that cannot satisfy a parent’s proof obligations.
8. Loop stops when the root closes or a budget (`max_input_tokens`, `max_output_tokens`, `max_attempts`) is exhausted.
