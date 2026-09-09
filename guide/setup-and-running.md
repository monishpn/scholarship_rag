# How to Run the Scholarship Chatbot

This guide explains how to set up and run the application on macOS and Windows.

## 1. Install Python

The application requires Python 3. Check whether it is already installed.

### macOS

```bash
python3 --version
```

### Windows

```powershell
python --version
```

If Python is missing, download it from [python.org/downloads](https://www.python.org/downloads/).

On macOS, download and run the `.pkg` installer, then open a new Terminal window and verify it with `python3 --version`.

On Windows, run the downloaded installer, select **Add Python to PATH**, choose **Install Now**, open a new PowerShell window, and verify it with `python --version`.

## 2. Open the project directory

Replace the example path with the actual location of the cloned repository.

### macOS

```bash
cd /path/to/scholarship_rag
```

### Windows

```powershell
cd C:\path\to\scholarship_rag
```

## 3. Create and activate a Python environment

A virtual environment keeps this project's dependencies separate from other Python projects.

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

After activation, the terminal normally displays `(.venv)` before the prompt.

## 4. Install the Python dependencies

Make sure the virtual environment is active.

### macOS

```bash
python3 -m pip install -r requirements.txt
```

### Windows

```powershell
python -m pip install -r requirements.txt
```

This installs `pypdf`, which the application uses to read the scholarship PDF.

## 5. Install Ollama

Ollama runs the language model that produces natural chatbot answers.

### macOS

Install Ollama from Terminal:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Or [download manually](https://ollama.com/download/Ollama.dmg).

Verify the installation:

```bash
ollama --version
```

### Windows

Install Ollama from PowerShell:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Or [download manually](https://ollama.com/download/OllamaSetup.exe).

Verify the installation:

```powershell
ollama --version
```

## 6. Download and verify the LLM model

The application uses `llama3.2:3b` by default. Installing Ollama does not automatically download this model.

Run this command on macOS or Windows:

```bash
ollama pull llama3.2:3b
```

The download may take some time. Confirm that the model is available:

```bash
ollama list
```

The output should include:

```text
llama3.2:3b
```

Keep Ollama running while using the chatbot.

## 7. Build the PDF index and start the application

For the first run, use `--reindex`. This reads the scholarship PDF, creates the searchable chunks in `data/index/chunks.json`, and starts the web application.

### macOS

```bash
cd /path/to/scholarship_rag
source .venv/bin/activate
python3 app.py --reindex
```

### Windows PowerShell

```powershell
cd C:\path\to\scholarship_rag
.venv\Scripts\Activate.ps1
python app.py --reindex
```

The terminal should display something similar to:

```text
Scholarship RAG ready at http://127.0.0.1:8000 (265 chunks)
```

Keep this terminal open while using the chatbot.

## 8. Open the frontend locally

There is no separate frontend command. `app.py` starts the backend API and serves the HTML, CSS, and JavaScript frontend together.

After starting `app.py`, open this address in a browser:

[http://localhost:8000](http://localhost:8000)

```text
python app.py --reindex
        ↓
Starts the backend API and serves the frontend
        ↓
Open http://localhost:8000
```

## 9. Run the application later

You only need `--reindex` when the PDF or indexing code changes. For normal later runs:

### macOS

```bash
cd /path/to/scholarship_rag
source .venv/bin/activate
python3 app.py
```

### Windows PowerShell

```powershell
cd C:\path\to\scholarship_rag
.venv\Scripts\Activate.ps1
python app.py
```

Then open [http://localhost:8000](http://localhost:8000). Make sure Ollama is running if you want LLM-generated answers.
