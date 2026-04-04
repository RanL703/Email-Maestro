# Product Requirements Document (PRD): Autonomous Executive Assistant Sandbox

**Target Deployment:** Hugging Face Spaces (Gradio UI + OpenEnv Container)
**Primary Dev Environment:** Kaggle / Jupyter Notebooks (`training_env.ipynb`)

---

## 1. Executive Summary
We are building a deterministic, isolated OpenEnv simulation of a corporate or academic workflow. Instead of wrapping a brittle, live API like Gmail (which causes rate limits and non-deterministic grading), we will engineer an **in-memory SQLite Mock Mail Server & Local File System**. 

The AI agent will act as an Autonomous Executive Assistant. It must navigate a chaotic mock inbox, extract deadlines to a mock task manager, negotiate meeting times, and perform Retrieval-Augmented Generation (RAG) over a mock file system to draft intelligent replies. 

This environment proves the agent's ability to act as a *router* and a *tool-user*, moving beyond text generation into full workflow automation.

---

## 2. Core Architecture & Stack
* **State Management:** In-memory SQLite (`sqlite3`) simulating a mail server, calendar, and file system.
* **Typing & Validation:** `pydantic` (Strictly defining Observations, Actions, and Rewards per OpenEnv spec).
* **Development & Debugging:** Kaggle/Jupyter Notebooks. The entire state machine, mock server, and OpenAI API baseline will be drafted and debugged interactively in a single notebook before containerization.
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
**Goal:** Prove the environment works using an LLM.
1. Write a `BaselineAgent` class in your notebook that takes your `OPENAI_API_KEY`.
2. Write a standard `while not done:` loop.
3. Pass the `WorkspaceObservation` to the LLM (using OpenAI structured outputs/function calling to ensure it returns the exact `AssistantAction` JSON).
4. Pass the LLM's action into the environment's `step()` function.
5. Print the interaction loop directly in the notebook to debug prompt formatting and reward shaping.

### Phase 5: Hugging Face Spaces & Gradio Deployment
**Goal:** Package the OpenEnv logic and build a visual interface so judges can physically see the agent working.

1. **The Gradio Wrapper (`app.py`):**
   * Build a simple Gradio UI with a Chatbot window for the agent's reasoning, and DataFrames/Tables visually representing the `Emails`, `Todos`, and `Files` SQLite tables.
   * As the OpenEnv `step()` function runs, update the Gradio state so the judges watch the inbox drain, the to-do list populate, and the replies send in real-time.
2. **Containerization (`Dockerfile`):**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   # OpenEnv requires specific metadata handling, Gradio runs on 7860
   EXPOSE 7860
   ENV GRADIO_SERVER_NAME="0.0.0.0"
   CMD ["python", "app.py"]
   ```
3. **OpenEnv Spec Compliance:** Ensure your `openenv.yaml` is correctly mapped to your Pydantic classes at the root of the repository.
4. **Push to HF:** Commit the repo to a Hugging Face Space, tag it with `openenv`, and ensure the Baseline script is easily executable via the README instructions.