# LLMprover

LLM-driven automatic theorem proving for Rocq/Coq, based on lemma-decomposition strategies.

## What works today

- Compile and validate `.v` scripts via `coqc`
- Multi-provider LLM client (OpenAI, Anthropic, Mistral)
- Domain types for goals, proof attempts, and proof records.
- Test suite with mocked LLM calls

## Roadmap

- [ ] LLM agent class (modes: direct, repair, decomposition, negation)
- [ ] Orchestration + strategies
- [ ] **`check_attempt` assumption check** — after a successful compile, verify that the main goal's proof depends only on lemmas listed in `new_lemmas` (e.g. via Rocq `Print Assumptions`); reject attempts that smuggle in extra dependencies
- [ ] **Library imports** — `Require Import` / prelude configuration so goals can use stdlib or project modules
- [ ] **Custom axiom environments** — goals stated over a user-provided context (axioms, existing lemmas), not only closed Prop/nat fragments
- [ ] Interactive proving mode with Pytanque

## Quick start

```bash
pip install -e ".[dev]"
python3 -m pytest
python main.py   # LLM generates a proof, coqc validates it (requires API key + coqc)
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
   ├── domain/          Goal, ProofAttempt, CoqcResult, ProofRecord (orchestrator & agent state, unused)
   ├── proof_script/    ProofScript (.v file I/O)
   ├── rocq/            CoqcBackend — assemble scripts, run coqc
   └── llm/             LLMClient — OpenAI, Anthropic, Mistral
```

**Dependency flow** (each layer only imports from layers above it):

```
proof_script  →  domain  →  rocq
                         →  llm
```

| Module | Role |
|--------|------|
| `proof_script.py` | Rocq source text (`ProofScript`, `from_file`) |
| `domain.py` | Core types: what to prove (`Goal`), what the LLM tried (`ProofAttempt`), validation outcome (`CoqcResult`). `ProofRecord` will hold the orchestrator’s global state — current proof progress and past search — but is not wired up yet. |
| `rocq.py` | Runs `coqc` on a `ProofScript` or a full `ProofAttempt` (with admitted lemmas) |
| `llm_client.py` | Stateless chat-completion wrapper over LLM provider SDKs |

**Planned** (not in the first release): `ProverAgent`, `Orchestrator`, `OrchestrationStrategy`, `AgentRegistry`, `pytanque_session`.

### Design notes

**`ProofScript`** is a thin dataclass around a `str` (Rocq source). A plain string would work for most of the current code; the class is kept as a hook for script-level operations (`from_file`, future `replace_lemma`, etc.). Whether a dedicated type pays off will become clearer once decomposition and merge are implemented — left as-is for now.

**`ProofRecord`** is defined in `domain.py` but not wired to any runner yet; it will hold orchestrator state when recursive search lands, and used by prover agents to recover search history.


**Decomposition invariant (not enforced yet).** When `new_lemmas` is non-empty, a valid parent proof should use *only* those admitted lemmas as logical dependencies — no hidden appeals to stdlib lemmas, axioms, or other admits. Today `check_attempt` only checks that the assembled script compiles; it does **not** yet verify assumptions. That check is required for sound recursive decomposition and is on the roadmap.

**Current limitations.** Scripts are self-contained fragments: no `Require Import`, no configurable prelude, and no support for proving goals in a rich ambient theory (custom axioms, imported libraries). The demo in `main.py` uses bare `Prop`/`/\` tactics without imports.

### Proof loop (target design)

1. `ProverAgent` proposes a `ProofAttempt` for a `Goal`.
2. `CoqcBackend.make_script` assembles a complete `.v` file and `check_attempt` validates it.
3. On failure, the raw `CoqcResult.output` is fed back into a `ProofRecord` for repair or mode switch.
4. The orchestrator updates a `ProofRecord` and keep a list of `ProofRecord` as search progresses (nodes, attempts, subgoals).
