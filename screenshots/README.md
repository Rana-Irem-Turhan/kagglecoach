# Screenshots

This folder is intentionally empty in the source distribution. Capture screenshots on your own machine and drop them here for inclusion in the README or docs.

## What's worth capturing

For a full README-quality set, four screenshots cover most of the story:

1. **`01-dialogue.png`** — the Streamlit UI showing the dialogue mid-flow (say, question 3 of 7, with the progress bar and the sidebar).
2. **`02-report-top.png`** — the top of a generated strategy report (User context, Dataset profile).
3. **`03-report-model-selection.png`** — the LLM-generated model selection section with visible source citations.
4. **`04-evidence.png`** — the retrieved evidence section at the bottom of the report showing which files were pulled from each collection.

Optional additions:

- **`05-azure-toggle.png`** — the sidebar with the Azure OpenAI toggle enabled and the "Active: azure" label.
- **`06-terminal.png`** — the indexer running in a PowerShell terminal (nice for showing the local-only workflow).

## Capturing on Windows

### Snipping Tool (recommended)

Windows 10 and 11 ship with a keyboard shortcut for the modern Snipping Tool:

```
Win + Shift + S
```

This dims the screen and lets you drag a rectangle. The screenshot copies to the clipboard and briefly appears as a notification; click the notification to open the editor, save as PNG.

For repeated captures, pin the Snipping Tool to the taskbar and use its Delay setting when you need to open menus that a screenshot would otherwise dismiss.

### Full-window capture

For a clean full-window screenshot without any surrounding chrome:

```
Alt + PrintScreen
```

Copies the active window to the clipboard. Paste into any image editor (Paint, Paint.NET, Photoshop) and export as PNG.

### Terminal / PowerShell capture

To capture a whole tall terminal window without cropping, resize the PowerShell window before the shot so all interesting content fits on one screen. Alt+PrintScreen then captures the entire window as one image.

## File naming

Prefer `NN-slug.png` naming (e.g., `01-dialogue.png`). Two-digit prefixes keep the folder listing in the intended order regardless of platform, and slug names make the README's image references self-documenting.

## Image size

Streamlit UIs look best at their native resolution — resist the urge to resize. If the file is too large for the repo, use `pngquant` or an online PNG compressor:

```powershell
# From a package manager:
winget install pngquant
pngquant --quality 65-90 --output 01-dialogue.min.png 01-dialogue.png
```

Typical target: 200-500 KB per screenshot. Larger is fine for docs, but avoid checking in files over 1 MB.

## `.gitignore` note

The repository's `.gitignore` explicitly does not ignore this folder. If you accidentally commit local paths or work-in-progress screenshots you don't want in the history, remove them with `git rm` before pushing.
