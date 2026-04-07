# Product Requirements Document (PRD): Autonomous Executive Assistant Sandbox

**Target Deployment:** Hugging Face Spaces (Gradio UI + OpenEnv Container)
**Primary Dev Environment:** Kaggle / Jupyter Notebooks (`training_env.ipynb`)

---

## Progress Note
Status as of 2026-04-08:

- The deterministic SQLite-backed workspace is implemented with action logging, seeded scenarios, snapshots, and richer step semantics.
- The OpenEnv contract is represented in typed Pydantic models for observations, actions, rewards, and policy decisions.
- Deterministic graders are implemented for all three seeded tasks with dense reward shaping and terminal success checks.
- A shared `EpisodeRunner` now owns the agent workflow loop across scripts, tests, the notebook, and Gradio.
- A deterministic baseline policy is implemented and solves all three seeded tasks end to end.
- An OpenRouter-backed `google/gemma-4-31b-it` policy path is integrated, prompt-hardened, and validated on the hard task.
- Separate app and training environments are in place, including a registered `scalerhack2-training` Jupyter kernel.
- The training notebook loads `.env.training`, exports traces, runs RL training, and saves checkpoints.
- A tabular Q-learning policy exists as a seeded-task RL prototype and can be trained, evaluated, and checkpointed.
- The current Gradio app can reset scenarios and run full episodes for baseline and OpenRouter policies.

Resume from here:

- Make the trained RL checkpoint a first-class runtime policy in the app and scripts.
- Refine the Gradio UI from one-shot episode execution into a stepwise or streaming judge-facing experience.
- Ensure the app, notebook, and scripts can all use the same trained RL artifact without drift.
- Expand notebook analysis cells and runtime metrics for stronger model-vs-baseline-vs-RL comparisons.
- Keep the current tabular RL policy as a prototype while leaving room for a richer learned policy after hackathon delivery.

---

## 1. Executive Summary
We are building a deterministic, isolated OpenEnv simulation of a corporate or academic workflow. Instead of wrapping a brittle, live API like Gmail (which causes rate limits and non-deterministic grading), we will engineer an **in-memory SQLite Mock Mail Server & Local File System**. 

The AI agent will act as an Autonomous Executive Assistant. It must navigate a chaotic mock inbox, extract deadlines to a mock task manager, negotiate meeting times, and perform Retrieval-Augmented Generation (RAG) over a mock file system to draft intelligent replies. 

This environment proves the agent's ability to act as a *router* and a *tool-user*, moving beyond text generation into full workflow automation.

---

## 2. Core Architecture & Stack
* **State Management:** In-memory SQLite (`sqlite3`) simulating a mail server, calendar, and file system.
* **Typing & Validation:** `pydantic` (Strictly defining Observations, Actions, and Rewards per OpenEnv spec).
* **Development & Debugging:** Jupyter Notebooks plus scriptable runners. The state machine, model prompts, rollout export, and RL smoke training are exercised from `training_env.ipynb` and mirrored by CLI scripts.
* **Model Runtime:** OpenRouter using `google/gemma-4-31b-it` for live policy inference, with prompt/schema hardening and response repair.
* **RL Prototype:** Tabular Q-learning over a finite action template catalog, with teacher warm-start from the deterministic baseline and JSON checkpoint persistence.
* **Deployment & Visualization:** Gradio (to visualize the inbox state for judges) packaged within a Docker container on Hugging Face Spaces.

---

## 3. Step-by-Step Implementation Plan

### Phase 1: The Mock Server Setup (Notebook Environment)
**Goal:** Build the deterministic world the agent will live in. Do this entirely in the first few cells of your Kaggle notebook so you can instantly query and reset the state.

1. **Database Initialization:** Create an in-memory SQLite database (`sqlite3.connect(':memory:')`).
2. **Table Creation:**
   * `Emails` (id, sender, recipient, subject, body, timestamp, is_read, is_archived)
   * `Todos` (id, task_name, deadline_date, context)
   * `Files` (id, filename, content_text) - *This acts as the local knowledge base.*
3. **The Wrapper Class (`MockWorkspace`):** Write Python methods to interact with this DB safely. 
   * `get_unread_emails()`
   * `send_reply(email_id, text)`
   * `create_todo(task, date)`
   * `search_documents(query)`

### Phase 2: OpenEnv Specifications (Pydantic Models)
**Goal:** Define the strict APIs the agent must use. This is the core of the hackathon requirement. 

**Observation Space:**
```python
class WorkspaceObservation(BaseModel):
    current_time: str
    unread_emails: List[Dict[str, str]] # ID, Sender, Subject snippet
    active_todos: List[str]
    last_action_status: str # e.g., "Email successfully sent to Manager"
```

**Action Space:**
```python
class AssistantAction(BaseModel):
    action_type: Literal["read_email", "reply", "forward", "add_todo", "archive", "search_files"]
    target_id: Optional[str] = None # email_id or file_id
    payload: Optional[str] = None # The body of the reply, or the search query
    secondary_payload: Optional[str] = None # Date for todos, or recipient for forwards
```

**Reward Space:**
```python
class TaskReward(BaseModel):
    step_reward: float
    total_score: float
    is_done: bool
    reasoning: str
```

### Phase 3: Task Definitions & Deterministic Graders
Implement the three required difficulty tiers. The grader simply runs SQL queries against your mock database to verify the agent's actions.

#### Task 1: Easy (Syllabus & Deadline Extraction)
* **Initial State:** DB injected with an email from `prof.smith@university.edu` containing 3 specific project deadlines.
* **Agent Goal:** Read email, create 3 corresponding tasks in the `Todos` table, and archive the email.
* **Grader Logic:** `SELECT COUNT(*) FROM Todos WHERE deadline_date IS NOT NULL;` -> If 3, return `+1.0`.

#### Task 2: Medium (Triage & Meeting Negotiation)
* **Initial State:** DB injected with 5 emails: 3 newsletters, 1 urgent client complaint, 1 team meeting reschedule request. 
* **Agent Goal:** Archive newsletters, forward the client complaint to `manager@company.com`, and reply to the reschedule request proposing a time.
* **Grader Logic:** Check if newsletters are marked `is_archived=True` (+0.3). Check if complaint is in the DB as sent to manager (+0.4). Check if reply contains a valid time string (+0.3).

#### Task 3: Hard (Autonomous RAG & Drafting)
* **Initial State:** DB injected with an email from a VIP stakeholder asking for specific metrics from the "Q3 Architecture Report".
* **Agent Goal:** Use `action_type: "search_files"` with query "Q3 Architecture", read the file contents, and use `action_type: "reply"` synthesizing the exact metrics from the file into a professional response.
* **Grader Logic:** Check if `search_files` was called (+0.3). Use regex to verify the specific metric string from the mock file exists in the sent reply body (+0.7).

### Phase 4: Baseline Agent Testing (Notebook Environment)
**Goal:** Prove the environment works using both a deterministic policy and a live model-backed policy.
1. Use the deterministic `BaselineAgent` to verify seeded tasks and grader behavior.
2. Use a standard `while not done:` loop, now centralized in `EpisodeRunner`.
3. Pass the `WorkspaceObservation` to the live model policy through OpenRouter using strict JSON outputs.
4. Pass the model action into the environment's `step()` function.
5. Print and export the interaction loop directly in the notebook to debug prompt formatting, policy behavior, and reward shaping.

#### Agent Workflow Loop
1. Load environment state
2. Generate observation
3. Send to LLM
4. Receive structured action
5. Execute action in workspace
6. Update state
7. Repeat until task complete

Implementation note: this loop is now represented directly in the shared `EpisodeRunner` so the notebook, scripts, tests, and Gradio app all execute the same control flow.

### Phase 5: Hugging Face Spaces & Gradio Deployment
**Goal:** Package the OpenEnv logic and build a visual interface so judges can physically see the agent working, including deterministic, model-backed, and learned-policy runs.

1. **The Gradio Wrapper (`app.py`):**
   * Build a Gradio UI that exposes selectable policies (`baseline`, `openrouter`, and trained `rl`) and visually represents the `Emails`, `Todos`, `Files`, and action history tables.
   * As the OpenEnv `step()` function runs, update the Gradio state step by step so judges can watch the inbox drain, the to-do list populate, and the replies send in real time.
   * Ensure the app can load the same trained RL checkpoint artifact produced by the notebook and CLI training scripts.
2. **Containerization (`Dockerfile`):**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.app.txt .
   RUN pip install --no-cache-dir -r requirements.app.txt
   COPY . .
   # OpenEnv requires specific metadata handling, Gradio runs on 7860
   EXPOSE 7860
   ENV GRADIO_SERVER_NAME="0.0.0.0"
   CMD ["python", "app.py"]
   ```
3. **OpenEnv Spec Compliance:** Ensure your `openenv.yaml` is correctly mapped to your Pydantic classes at the root of the repository.
4. **Push to HF:** Commit the repo to a Hugging Face Space, tag it with `openenv`, and ensure the policy runners and training instructions are easily executable via the README instructions.
