# KaggleCoach setup

This guide targets **Windows 10 and 11**. Linux and macOS notes are at the bottom, but the reference environment is Windows because that is what Microsoft's Foundry Local summer school samples target.

## Prerequisites

Open **PowerShell** (not Command Prompt). All commands below assume PowerShell.

### 1. Foundry Local

```powershell
winget install Microsoft.FoundryLocal
```

After install, restart PowerShell so the `foundry` command is on your PATH. Confirm:

```powershell
foundry --version
foundry model list
```

The `model list` should show at least a chat model (`phi-3.5-mini` or similar) and an embedding model (`qwen3-embedding-0.6b` or similar). If the exact model names differ, edit `config.toml` to match your local catalog before running the indexer.

### 2. Python 3.11 or 3.12

```powershell
winget install Python.Python.3.12
```

Restart PowerShell, then:

```powershell
python --version
```

should print `Python 3.12.x`. If not, check `Get-Command python` — you may have an older version earlier on PATH.

### 3. Git (only if cloning from a repository)

```powershell
winget install Git.Git
```

## Install KaggleCoach

```powershell
# Get the source
git clone <repo-url> kagglecoach
cd kagglecoach

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
# If activation is blocked by execution policy:
#     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# and then rerun Activate.ps1.

# Install Python dependencies
pip install -r requirements.txt
```

If you unzipped an archive instead of cloning, `cd` into the extracted `kagglecoach` folder and skip the `git clone` step.

## Build the FAISS index

The three FAISS collections (`tabular`, `nlp`, `general_ml`) must be built once before the app can retrieve evidence. This takes 30–90 seconds depending on Foundry Local's cold-start time.

```powershell
python -m kagglecoach.indexer
```

Expected output (abbreviated):

```
[tabular]  scanning ...\knowledge\tabular
  01-lightgbm-strategy.md                     →   6 chunk(s)
  ...
[nlp]  scanning ...\knowledge\nlp
  ...
[general_ml]  scanning ...\knowledge\general_ml
  ...
[tabular]    27 chunk(s) from 5 file(s), avg 900 chars, dim 768
[nlp]        15 chunk(s) from 3 file(s), avg 900 chars, dim 768
[general_ml] 22 chunk(s) from 4 file(s), avg 900 chars, dim 768
```

If the counts are 0, either Foundry Local isn't running or the model aliases in `config.toml` don't match the ones on your machine. Run `foundry model list` and edit `config.toml` accordingly.

To rebuild only one collection after editing its knowledge files:

```powershell
python -m kagglecoach.indexer --collection tabular
```

## Run the app

```powershell
streamlit run kagglecoach\ui.py
```

Streamlit opens `http://localhost:8501` in your default browser. The first response takes longer than subsequent ones because Foundry Local downloads/loads the chat model on first use.

### Change the port

```powershell
streamlit run kagglecoach\ui.py --server.port 8600
```

## Optional: Azure OpenAI mode

KaggleCoach can route chat calls to Azure OpenAI while keeping embeddings on Foundry Local. Enable it via the sidebar toggle at runtime — no config change needed to try it — but the env vars must be set before launching Streamlit:

```powershell
# Add to your PowerShell profile or set per-session:
$env:AZURE_OPENAI_ENDPOINT = "https://<your-resource>.openai.azure.com"
$env:AZURE_OPENAI_API_KEY = "<your-key>"
$env:AZURE_OPENAI_DEPLOYMENT = "<your-deployment-name>"
```

For persistence across shells:

```powershell
[System.Environment]::SetEnvironmentVariable("AZURE_OPENAI_ENDPOINT", "...", "User")
[System.Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_KEY", "...", "User")
[System.Environment]::SetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT", "...", "User")
```

Close and reopen PowerShell after setting user-level variables.

**Privacy note.** Even in Azure mode, only the chat prompt (dialogue answers + retrieved evidence excerpts) goes to Azure. The uploaded CSV never leaves your machine — EDA runs in the local Python process and only summary statistics are inlined into the prompt.

## Verify the install

Fast smoke test that doesn't need Foundry Local running:

```powershell
python -m pytest tests -q
```

You should see `89 passed` in about a second. This is a good check that FAISS installed correctly and no Python module is broken.

## Troubleshooting

### `pip install faiss-cpu` fails on Windows

FAISS wheels are officially published for Windows only for Python 3.8-3.12. If you're on 3.13 or later, downgrade to 3.12:

```powershell
winget install Python.Python.3.12
```

and rebuild the venv with `python3.12 -m venv venv`.

### `foundry_local` module not found

The Python SDK is a separate package. It is included in `requirements.txt` as `foundry-local-sdk`. If pip couldn't install it (e.g., you're offline), the app still starts but any chat call will raise a clear error.

### PowerShell blocks `Activate.ps1`

Windows blocks unsigned local scripts by default:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run `.\venv\Scripts\Activate.ps1`.

### Streamlit page keeps auto-refreshing

Disable Streamlit's file watcher if you're editing files while the app runs:

```powershell
streamlit run kagglecoach\ui.py --server.runOnSave false
```

### Confusing "FAISS index is empty" error

Run `python -m kagglecoach.indexer` at least once. If already run, verify `data/faiss/*.faiss` files exist. Delete `data/` and reindex if they're corrupt.

## Linux / macOS notes

The install steps are structurally identical:

```bash
# Linux
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m kagglecoach.indexer
streamlit run kagglecoach/ui.py

# macOS
brew install python@3.12
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m kagglecoach.indexer
streamlit run kagglecoach/ui.py
```

Foundry Local's Linux/macOS support is still evolving. Check <https://learn.microsoft.com/azure/foundry-local/> for the current install method on non-Windows platforms. Everything else in KaggleCoach works identically.
