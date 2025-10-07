NetIntelli X

Project Overview
- NetIntelli X is a Python desktop app for multi-vendor network automation with a built-in terminal, AI-assisted command generation, and device context awareness.
- Uses `Netmiko` for SSH and supports AI providers including Gemini, OpenAI, and local `Ollama` models.
- The AI assistant now responds as a seasoned network engineer with multi-vendor experience and safety-first guidance.

Key Features
- Multi-vendor SSH connectivity: H3C/Huawei, Cisco, Juniper, Arista, and others.
- Context-aware AI: Considers manufacturer, model, version, and live running config.
- Interactive terminal: Real-time output, manual input, direct typing support.
- Profiles: Save/load connection details and AI context.
- Local or cloud AI: Choose Gemini, OpenAI, or local Ollama.

Latest Changes
- Updated AI persona to an experienced, helpful multi-vendor network engineer.
- Chat workflow:
  - Per-command `Send to CLI` buttons inside chat results.
  - `Sync Commands` button to send the last AI response sequentially.
  - Chat results are no longer mirrored into the AI push pane.
- AI Assistant push button removed: “Push to connected device” is no longer present in the AI output panel.
- Vendor-aware device controls added near the terminal:
  - `Enter Privileged/Config Mode`: enters mode based on manufacturer.
  - `Save Config`: executes vendor-specific save/commit commands.
  - `Quit`: steps back one mode level with vendor-aware `quit`/`exit`.

Vendor Command Mappings
- Enter Privileged/Config Mode:
  - Cisco: `enable` then send Enable password.
  - H3C/Huawei (VRP): `system-view`.
  - Juniper: `configure`.
  - Fallback: `enable` (optional password) then `system-view`.
- Save Config:
  - Cisco: `write memory`.
  - H3C/Huawei: `save` followed by auto-confirm `Y`.
  - Juniper: `commit`.
  - Fallback: `save` with auto-confirm `Y`.
- Quit one level:
  - H3C/Huawei: `quit`.
  - Cisco/Juniper/others: `exit`.

Quick Start
- Requirements: Windows, Python 3.x, internet access for cloud AI (optional), local Ollama if using local models.
- Run: double-click `run_app.bat` (or run `python app_gui.py`).
- Connect:
  - Fill `Manufacturer`, host/IP, username, password, and optional `Enable password`.
  - Click `Connect`.
- Fetch context:
  - Use `Fetch Running Config` to pull live config for AI context.
- Use AI:
  - Go to the Chat pane, enter your request, choose provider/model, and submit.
  - Click per-command `Send to CLI` or use `Sync Commands` to send all.
- Device controls:
  - Click `Enter Privileged/Config Mode` before making changes.
  - Use `Quit` to step back one mode level.
  - Click `Save Config` to persist changes.

Notes and Tips
- Set `Manufacturer` correctly; all vendor-aware actions depend on it.
- Some devices prompt for pager control; the terminal auto-sends space when `--More--` is detected.
- At top-level on Cisco, `exit` may log you out; use `Quit` thoughtfully.
- For H3C/Huawei, `Save Config` auto-confirms `Y`.

Troubleshooting
- `ModuleNotFoundError: No module named 'telnetlib'`: Python 3.13 removed `telnetlib`; ensure `netmiko` is up to date or use Python < 3.13.
- Ollama connection issues: ensure `http://localhost:11434` is running and models are pulled.
- Incorrect vendor commands: verify `Manufacturer` is set and provide clear requests.
- GUI freezes: long operations may block; consider patience for fetch/AI steps.

Security
- `profiles.json` stores connection info locally. Handle your machine securely.
- API keys for cloud AI are not persisted across sessions.

Contributing
- Issues and PRs are welcome. Please keep changes focused and consistent with existing style.