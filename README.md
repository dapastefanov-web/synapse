Here is the comprehensive, highly detailed, and visually structured `README.md` for Synapse. This is designed to serve as both a compelling landing page for your GitHub repository and a complete technical manual for power users.

---

# 🧠 Synapse

### The Configurable, Polyglot AI Engineering Swarm

*Synapse is a terminal-native, configuration-driven multi-agent framework. It transforms your CLI into a localized "System-of-Intelligence" capable of planning, executing, testing, and debugging software autonomously inside isolated, dynamic DevContainers.*

[Features](https://www.google.com/search?q=%23-core-features) • [Installation](https://www.google.com/search?q=%23-installation) • [Quick Start](https://www.google.com/search?q=%23-quick-start) • [The Stacks](https://www.google.com/search?q=%23-the-tri-stack-architecture) • [Customization](https://www.google.com/search?q=%23-the-yaml-configuration-engine) • [Security](https://www.google.com/search?q=%23-security--the-internal-jail)

---

## 🚀 Why Synapse?

Standard AI coding assistants act as intelligent autocomplete or passive chatbots. They require you to manage the environment, run the tests, and paste the error logs back into the prompt.

**Synapse is different.** It is an orchestration engine. By combining the state-machine logic of **LangGraph**, the provider-agnostic routing of **LiteLLM**, and the execution safety of **Docker**, Synapse delegates tasks to a swarm of specialized AI roles (Architect, Coder, Debugger, Summariser). It edits your code, runs your tests inside an isolated polyglot container, reads the stack traces, and self-corrects—all before prompting you for final approval.

---

## ✨ Core Features

* **Multi-Provider Optimization:** Route planning tasks to ZhipuAI, rapid coding to Groq, and deep debugging to NVIDIA NIM or Together AI. Never get bottlenecked by a single provider's rate limits.
* **Dynamic Polyglot Sandboxing:** Instantly provisions isolated environments using Nix and Docker. Whether you are writing Rust, Python, PHP, or TypeScript, Synapse dynamically installs the compilers it needs.
* **Model Context Protocol (MCP) Ready:** Natively connect to any official MCP server (e.g., Sequential Thinking, Postgres tools) with zero extra code.
* **Semantic Vector Memory:** Built-in zero-dependency vector search (`sqlite-vec`). The AI instantly finds relevant codebase context without blowing up your token limits.
* **Advanced Command Lexer:** Chain commands synchronously (`&&`), bypass menus dynamically (`*`), and expand complex prompts via custom macros (`@`).

---

## 📦 Installation

### Prerequisites

1. **Python 3.12+** installed on your host machine.
2. **Docker Daemon** installed and running.

### Standard Installation (Source)

```bash
git clone https://github.com/yourusername/synapse-v2.git
cd synapse-v2
pip install -e .

```

### Arch Linux (AUR)

*(Coming soon)*

```bash
yay -S synapse-ai-cli

```

---

## ⚡ Quick Start: Your First Agentic Loop

Synapse operates on a **VS-Code-like DevContainer** philosophy. You initialize Synapse on a per-project basis.

### 1. Initialize the Environment

Navigate to an existing project on your machine (or create a new folder) and run:

```bash
cd /path/to/my-project
synapse /init-docker

```

*What this does:* Synapse creates a `.synapse/` directory in your project containing a dynamic `Dockerfile` and a `shell.nix` file. It then builds a lightweight, isolated execution environment specifically for this codebase.

### 2. Configure Your API Keys

Check which providers are supported and configure your keys:

```bash
synapse /api

```

Add your preferred keys (e.g., `GROQ_API_KEY`, `NVIDIA_NIM_API_KEY`) to your global `~/.bashrc` or export them directly.

### 3. Run a Command

Start Synapse and issue a complex chained command:

```bash
synapse
> /stack *autonomous && @arch Build a new FastAPI authentication router and ensure it is fully tested.

```

---

## 🧠 The Tri-Stack Architecture

Synapse allows you to dynamically switch the LangGraph topology based on the complexity of your task using the `/stack` command.

### 1. The Fast Stack (`/stack *fast`)

**Best for:** Typos, quick refactors, docstrings, and single-file changes.

* **Flow:** User Prompt ➡️ Groq (Coder) ➡️ Write File ➡️ NVIDIA NIM (Summariser) ➡️ Done.
* **Speed:** Extremely low latency. No testing or debugging phases.

### 2. The Balanced Stack (`/stack *balanced`)

**Best for:** Standard feature implementation requiring structural thought but no active execution.

* **Flow:** Architect (Splits task) ➡️ Coder (Implements) ➡️ Static Debugger (Reads code to enforce conventions without running tests) ➡️ Summariser.

### 3. The Autonomous Stack (`/stack *autonomous`)

**Best for:** Deep architectural refactoring, multi-file generation, and Test-Driven Development (TDD).

* **Flow:** The ZhipuAI Architect breaks your prompt into distinct tasks. It spawns **parallel Map-Reduce execution branches**. Moonshot Kimi writes the code, while DeepSeek V4 runs your test suite inside the Docker container. If the tests fail, they loop and fix it until `SUCCESS` is achieved. The engine then intelligently merges the parallel patches Git-style.

---

## ⌨️ Shell Mastery: Lexer & Access Control

Synapse's prompt isn't just a chatbox; it is a programmable shell interface.

### Synchronous Chaining (`&&`)

Queue up system commands and LLM prompts in a single line. They will execute sequentially.

```text
> /stack *balanced && /init-docker && Write a README for this project.

```

### Inline Arguments (`*`)

Bypass interactive terminal menus to maintain your flow state.

```text
> /access *trust

```

### Custom Macros (`@`)

Define macros in your `cli_config.yaml` to expand shorthand into complex, highly specific system prompts.

```text
> @test the user_service.py file
# Expands to: "Ensure you write comprehensive unit tests using pytest with high coverage for... the user_service.py file"

```

### Access Control & Human-in-the-Loop (`/access`)

You control exactly when the AI is allowed to touch your bare-metal files.

* `/access *trust`: The AI loops run, tests pass, and the files are overwritten automatically. You get a summary.
* `/access *no_trust`: The AI finishes its work and pauses. You are presented with a unified diff and a prompt: `Accept / Decline / Rewrite`. Choosing **Rewrite** allows you to type a critique, sending the graph backward to fix its logic before applying.

---

## ⚙️ The YAML Configuration Engine

Synapse is built to be customized. Core logic is not hardcoded; it is dictated by cascading YAML files.

* **Global Config:** `~/.config/synapse/` (Applies to all projects)
* **Local Override:** `[Your Project]/.synapse/config.yaml` (Overrides global for specific projects)

### 1. `agents.yaml`

Swap models, change providers, or edit system prompts in seconds.

```yaml
agents:
  architect_autonomous:
    provider: zhipuai
    model: glm-5.1
    system_prompt: "You are the Lead Systems Architect. Split the request into independent Tasks."
    tools: [firecrawl, vector_gatherer, mcp_sequential_thinking]

```

### 2. `pipeline.yaml`

Modify the literal edges of the LangGraph state machine. Want to add a "Security Auditor" agent before the Summariser? Just define the agent and wire it up here.

### 3. `cli_config.yaml`

Define your CLI aliases and macros.

```yaml
aliases:
  "/init": "/init-docker"
macros:
  "@arch": "System Directive: Prioritize architectural patterns. Analyze the following: "

```

---

## 🛠️ Extensibility: Custom Tools & MCP

Synapse provides three tiers of tooling to give your agents maximum leverage.

1. **Native Tools:** File read/write, web search via Firecrawl, and Semantic Codebase Search via `sqlite-vec` are built-in.
2. **Dynamic Python Tools:** Drop any `.py` script into `.synapse/tools/`. Synapse dynamically parses your Python type-hints to generate the required LLM JSON schema automatically. (You can also drop a `.yaml` file of the same name to manually override the schema).
3. **Model Context Protocol (MCP):** Connect to the global MCP ecosystem seamlessly. Define the server in your config:

```yaml
mcp_servers:
  postgres_manager:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]

```

Synapse will automatically negotiate with the MCP server and inject its tools into the agent's context.

---

## 🛡️ Security & The Internal Jail

Executing AI-generated code on your host machine is dangerous. Synapse solves this via a robust, two-tiered security model.

**1. Polyglot Isolation**
Code and tests are *never* run on your host shell. They are executed via the Docker Python SDK inside the dynamic DevContainer built by `/init-docker`.

**2. The Discretionary Access Control (DAC) Jail**
To test code properly, the AI needs access to your API keys (e.g., Stripe, AWS).

* You place your `.env` or `bashrc` with testing keys inside your project's `.synapse/` folder.
* Inside the Docker container, the LLM executes file exploration and coding as `ai_user`. Linux file permissions (`chmod 700`) completely block the AI from reading the `.synapse` folder, meaning it cannot hallucinate a command to steal or leak your raw keys.
* When the Debugger triggers the test suite, Synapse executes the test command strictly as `test_user`, who *is* allowed to source the keys. The tests run securely, and only the sanitized stack trace is returned to the LLM.

---

## 🤝 Contributing

Synapse is an open-source framework designed to evolve. We welcome pull requests for:

* New native tools.
* Advanced LangGraph sub-graph templates.
* Integrations with new MCP servers.
* UI enhancements for the `rich` terminal dashboard.

Please read our `CONTRIBUTING.md` for guidelines on setting up your local development environment.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.