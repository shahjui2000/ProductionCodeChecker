# Production Code Checker Agent 🕵️‍♂️

An AI-powered CLI tool that reviews Python code for production readiness. It acts as a **Production ML Engineering Assistant**, analyzing your code for reliability, maintainability, and scalability gaps.

## 🚀 Features

The agent reviews code through the lens of a Senior ML Engineer, checking for:

1.  **Error Handling & Resilience**: Try-catch blocks, OOM protection, timeouts.
2.  **Observability & Monitoring**: Logging standards, metrics tracking.
3.  **Configuration Management**: No hardcoded credentials/hyperparams.
4.  **Resource Management**: GPU memory cleanup, connection pooling.
5.  **Testing & Validation**: Unit tests, input validation.
6.  **Code Quality**: Type hints, docstrings, DRY validation.
7.  **Security**: Input sanitization, secrets management.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone git@github.com:shahjui2000/ProductionCodeChecker.git
    cd ProductionCodeChecker
    ```

2.  **Set up a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

1.  Get a free **Hugging Face Token** from [here](https://huggingface.co/settings/tokens).
2.  Create a `.env` file in the root directory:
    ```bash
    cp .env.example .env
    ```
3.  Add your token to `.env`:
    ```ini
    HF_TOKEN=hf_YourTokenHere
    ```

## 📖 Usage

Run the agent on any Python file:

```bash
python3 main.py path/to/your/script.py
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Hugging Face Model ID to use | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| `--api-key` | Pass HF Token directly (overrides .env) | `None` |

## 📊 Example

Analyze the included test file:

```bash
python3 main.py test_bad_code.py
```

**Output Snippet:**

```markdown
### 7. Security
- **Issue**: Hardcoded AWS credentials.
- **Severity**: Critical
- **Suggested Fix**: Use environment variables for AWS credentials.
```

## 📄 License

MIT
