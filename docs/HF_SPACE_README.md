---
title: EmailMaestro | Executive Assistant Sandbox
emoji: "🧭"
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - docker
  - gradio
short_description: OpenEnv executive assistant sandbox demo for judges.
---

# Project Epsilon

Welcome judges of the **OpenEnv Scaler x Meta x PyTorch Hack**. This Space hosts **EmailMaestro**, our deterministic executive assistant environment and policy demo built by **Team Epsilon** for repeatable agent evaluation, visible tool use, and side-by-side policy comparison.

## Team Epsilon Roster

- Team name: `Team Epsilon`
- Hugging Face Space: `Flickinshots/EmailMaestro`
- Live app: `https://Flickinshots-EmailMaestro.hf.space`
- Public repository view on Hugging Face: `https://huggingface.co/spaces/Flickinshots/EmailMaestro`

- `@flickinshots` — Team lead and primary Space owner
- `@ShreyaKhatik` — Team member
- `@itsayushdey` — Team member

## Executive Summary

EmailMaestro is an **Autonomous Executive Assistant Sandbox** designed around the OpenEnv pattern: typed observations, typed actions, deterministic rewards, and a visible environment loop. Instead of depending on a brittle live email provider, the agent operates inside an isolated SQLite-backed workspace that simulates an inbox, a todo manager, and a local document store. That lets judges inspect policy quality through reproducible runs rather than through one-off anecdotal chats.

This Space is intended to show three things clearly:

- The agent can operate as a structured tool user, not just a text generator.
- The same environment loop is shared across notebook experiments, CLI evaluation, tests, and the live Gradio UI.
- Baseline, learned, and model-backed policies can be compared under the same task and reward conditions.

## Why This Fits The Hackathon

OpenEnv was introduced by Hugging Face and Meta as an open framework for building agent environments with typed observations, actions, and rewards. The Scaler hack timeline for this event lists the main build window as **March 25, 2026 through April 8, 2026**, with finals on **April 25-26, 2026** in Bengaluru. Our submission is shaped directly around that evaluation style:

- Deterministic environment setup
- Typed environment contracts
- Observable step-by-step policy execution
- Reproducible seeded tasks
- Judge-friendly visualization of state transitions

## Problem Framing

Most assistant demos stop at text quality. We wanted to show that an agent can manage a workflow end to end:

- read a chaotic inbox
- extract structured work into a task list
- triage low-priority versus high-priority communication
- search a local knowledge source before replying
- produce actions that can be graded deterministically

To make that possible under hackathon constraints, we replaced live services with a controlled mock workspace that still feels operationally realistic.

## Core Architecture

- **Environment state:** in-memory SQLite workspace simulating emails, todos, files, and action history
- **OpenEnv contract:** Pydantic models defining observations, actions, rewards, and policy decisions
- **Execution loop:** shared `EpisodeRunner` used by tests, scripts, notebook experiments, and the Gradio app
- **App policies:** deterministic baseline and an RL mode where OpenRouter Gemma generates actions using the tabular RL checkpoint recommendation as guidance
- **Validator inference:** OpenRouter-backed Gemma execution through the OpenAI client compatibility layer
- **UI layer:** Gradio control room plus visible workspace snapshots for judges

## Seeded Judge Tasks

### 1. Easy: Deadline Extraction

The environment injects an academic email containing multiple deadlines. The policy must read the message, create the correct todo entries, and archive the source email.

### 2. Medium: Inbox Triage And Negotiation

The environment mixes newsletters, an urgent complaint, and a meeting reschedule request. The policy must archive low-value mail, escalate the complaint properly, and send a concrete meeting reply.

### 3. Hard: RAG Reply

The environment includes a stakeholder email asking for exact metrics from a local report. The policy must search the file store, recover the relevant values, and draft a grounded reply using the retrieved evidence.

## What Judges Can Inspect In This Space

- Live observation payloads
- Workspace tables for emails, todos, files, and action logs
- Step-by-step trace rows with reasoning, action type, status, score, and done state
- Differences between `baseline` and the OpenRouter-guided bundled `rl` checkpoint policy

## Runtime And Deployment Notes

- SDK: `docker`
- App port: `7860`
- Entry point: `python app.py`
- Optional secret: `OPENROUTER_API_KEY`
- OpenAI-compatible base URL: `https://openrouter.ai/api/v1`
- Model: `google/gemma-4-31b-it`
- Bundled RL checkpoint path: `artifacts/checkpoints/q_policy_notebook.json`
- The Space is deployed from the same repository used for local tests and notebook-backed experiments

## Recommended Judge Flow

1. Open the Space and choose one of the seeded scenarios.
2. Run the deterministic `baseline` policy for a guaranteed reference trace.
3. Switch to `rl` so Gemma receives the learned checkpoint recommendation and generates the runtime action.
4. Compare how the workspace mutates after each step instead of evaluating only the final response.
5. Use the root `inference.py` path for OpenRouter-backed Gemma evaluation when the validator runs model inference.

## Implementation Notes

- The app, scripts, notebook, and tests all rely on the same `EpisodeRunner` workflow loop.
- Live API access stays in the policy layer, so deterministic evaluation remains possible without network access.
- The current RL path is intentionally lightweight and reproducible: a tabular Q-learning prototype trained over seeded action templates.
- The Gradio interface is designed for demonstration and debugging, not just for final-state screenshots.

## What We Want Judges To Notice

- Strong separation between environment state, policy choice, and reward logic
- Clear evidence of agent tool use
- Reproducibility across runs
- A hackathon-friendly deployment that still preserves engineering discipline

## References And Context

- Hack dashboard: https://www.scaler.com/openenv-hackathon
- OpenEnv launch: https://huggingface.co/blog/openenv
