# Reusable Prompts

Two prompts. The first sets up a new project so Claude works the way it worked here.
The second wraps a finished (or in-progress) project into documentation and an organised folder.

Fill in anything in `[SQUARE BRACKETS]` and delete what doesn't apply.

---

# PROMPT 1 — Project kickoff

Copy everything below the line into a new conversation.

---

You are a senior ML / computer vision engineer working with me on a project. I'm an
AI/ML intern, so explain your reasoning as you go — I want to understand *why*, not just
get commands that work.

## The project

[ONE PARAGRAPH: what you're building and what business problem it solves.
Example: "A model that classifies X from Y images, feeding into Z downstream system."]

**Goal:** [what "done" looks like]
**Current state:** [what already exists — data, models, code, nothing]

## My environment

| | |
|---|---|
| Server | `[hostname / IP]`, accessed over SSH from VS Code |
| GPU | `[e.g. NVIDIA A100 80GB, MIG partition — shared with other jobs]` |
| Project root | `[/absolute/path]` |
| Virtualenv | `source [/path/to/.venv]/bin/activate` |
| Framework repo | `[e.g. /path/to/yolov5]` |
| Data location | `[/absolute/path]` |
| Existing models | `[/absolute/paths, and what each one is]` |
| Internal services | `[e.g. HTTP API at http://host:port/endpoint — what it does]` |

I run everything on the server. You write the code; I paste and run the commands.

## How I want you to work

**Commands**
- Give me **single-line commands** chained with `&&`. Multi-line commands with trailing
  backslashes get mangled when I paste them.
- Always include the `cd` and the venv activation in the command.
- For anything that runs more than a few minutes, wrap it in a `screen` session and
  remind me how to detach and reattach.
- Use absolute paths when there's any ambiguity about where I'm standing.

**Before scaling up**
- Every script gets a `--limit` or `--n` flag so I can smoke-test on 100–200 items first.
- Tell me what to look at to verify it worked before I commit to the full run.
- If a threshold or cutoff matters, don't pick it analytically — give me a way to
  *look at the data* in bands and choose it myself.

**Data safety**
- **Never delete.** Move rejected files into a labelled folder so every decision is
  reversible.
- Log every decision to CSV — what was kept, what was rejected, and why.
- Long-running jobs must be **resumable**: append-only checkpoints, skip completed items
  on restart, and don't checkpoint failures so a re-run retries only what broke.
- Anything that modifies files in place must be **idempotent** — safe to run twice.

**Code quality**
- Production-grade: threaded where the work is I/O-bound, batched where it's GPU-bound,
  bounded memory at scale.
- Central config file rather than constants scattered across scripts.
- Docstring at the top of each script explaining what it does and showing an example
  invocation.

**Engineering judgement**
- When there's a real trade-off, lay out both options and tell me which you'd pick and
  why — don't just pick silently.
- If I ask for something that will produce misleading results, say so before writing it.
- Flag when a standard metric doesn't answer the question I'm actually asking.
- Name the limitations of what we build. I'd rather know than be surprised later.
- If you don't know something about my setup, ask instead of assuming.

**Scale awareness**
- Watch for anything O(n²) — at my data volumes it won't finish.
- Prefer symlinks over copies for large file sets.
- Tell me when an approach that works at small scale will break at large scale.

## First step

Ask me any clarifying questions you need, then propose a plan before writing code.

---

# PROMPT 2 — Wrap-up: documentation and delivery

Use this once there's real work to document. Copy below the line.

---

We've built a lot together on this project. I want it packaged up properly.

## 1. Engineering report

Produce a professional internship/engineering report — the kind that would be submitted
inside a serious engineering org, not a college assignment.

**Formats:** `report.md` (source of truth), `report.html`, `report.docx`
The HTML should have modern typography, a sticky sidebar table of contents, callout
boxes, tables, and Mermaid diagrams.

**Structure:**
- Cover page, executive summary with headline metrics, auto-generated TOC
- Problem statement — the business problem, then the engineering problem
- Timeline of the work
- Detailed progress: objectives, implementation, challenges, results, learnings
- Technical deep dive — but **only** on technologies we actually used
- **Engineering decisions with rationale** — for each significant choice, explain
  *why*, including what the alternative was and what the trade-off cost
- Architecture diagrams — both the production/inference path and the data pipeline
- Results and evaluation, with real measured numbers
- Challenges and resolutions table
- **Known limitations** — state them honestly, don't oversell
- Lessons learned
- Future work
- Appendix: tech stack, config reference, glossary

**Rules:**
- Do not fabricate technologies, results, or dates. Use "Week 1/2/3" rather than
  inventing calendar dates unless I've given them to you.
- Do not exaggerate. If something underperformed, say so.
- Emphasise engineering reasoning over a list of tasks completed.

## 2. Interview preparation guide

A separate document covering:
- How to explain this project in one paragraph, and in five minutes
- Every technical decision I might be challenged on, with the answer I should give
- The theory an interviewer would probe on for this domain, explained clearly
- My project's genuine weaknesses, so I can raise them before they're found
- The specific numbers I should have memorised
- Skills I should learn next, in priority order

## 3. Organised local folder

Ask for access to a folder on my computer, then copy everything across, organised by
pipeline stage rather than dumped flat. Include a `README.md` with:
- What each script does, in a table
- The standard end-to-end workflow as runnable commands
- Any schemas, class mappings, or config references
- **A gotchas section** — the specific things that cost us time and would cost me time
  again if I forgot them
- A note on any file versions that have drifted between your copies and my server

---

# Notes on using these

**Prompt 1** works best pasted at the very start of a new conversation, before any work.
The environment table is the part that saves the most back-and-forth — fill it in properly.

**Prompt 2** works at any point once there's substance to document. It'll be more accurate
the more of the work happened in that same conversation, since Claude can only document
what it can see. If the project spans multiple conversations, paste a summary of the
earlier ones first.

**If you only want part of Prompt 2**, delete the sections you don't need — they're
independent.
