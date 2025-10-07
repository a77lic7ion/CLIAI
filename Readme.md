NetIntelli X

Project Overview
- NetIntelli X is a Python desktop app for multi-vendor network automation with a built-in terminal, AI-assisted command generation, and device context awareness.
- Supports Serial, SSH, and Telnet connections with a unified terminal interface.
- Works with AI providers including Gemini, OpenAI, and local `Ollama` models.

Key Features
- Multi-vendor connectivity: Serial/SSH/Telnet for H3C/Huawei, Cisco, Juniper, Arista, and others.
- Context-aware AI: Considers manufacturer, model, version, and live running config.
- Interactive terminal: Real-time output, manual input, direct typing and Enter-to-send.
- Profiles: Save/load connection details and AI context, including network fields.
- Local or cloud AI: Choose Gemini, OpenAI, or local Ollama.

Latest Changes
- Connection panel updates:
  - `Connection Type` selector: `Serial`, `SSH`, `Telnet`.
  - `Host` and `Port` fields added for network connections.
  - Auto-set default ports when switching type: SSH `22`, Telnet `23`; clears for Serial.
- Serial/SSH/Telnet adapters:
  - Normalized write/read interface for all connection types.
  - SSH via `Netmiko` (device type derived from `Manufacturer`).
  - Telnet via `telnetlib` (if available).
- Terminal input improvements:
  - New `?` button sends a space + `?` + Return to reveal available commands.
  - Pressing `Enter` in the inline input sends the typed command immediately.
- Profiles persistence:
  - Saves and restores `Connection Type`, `Host`, and `Port` alongside existing fields.
  - Reset to `Serial` and clears `Host`/`Port` on profile delete.
- Command DB build:
  - Adds a new `command_tree` table storing hierarchical command paths and raw `?` output.
  - BFS expansion of commands via CLI `?`, with safe depth/branch limits.
  - Continues to collect AI guidance for major categories (VLANs, Interfaces, etc.).

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

Connection Types
- Serial:
  - Select COM port and baud rate.
  - Username/password fields are optional and not used by Serial.
- SSH:
  - Set `Host`, `Port` (default `22`), `Username`, `Password`, and optional `Enable password`.
  - Device type derived from `Manufacturer` for `Netmiko` (`cisco_ios`, `hp_comware`, `juniper_junos`, `arista_eos`).
- Telnet:
  - Set `Host`, `Port` (default `23`).
  - Requires `telnetlib`; note Python 3.13 removed it.

Quick Start
- Requirements: Windows, Python 3.10–3.13, internet access for cloud AI (optional), local Ollama if using local models.
- Install dependencies: `pip install netmiko pyserial`
- Run: double-click `run_app.bat` or run `python app_gui.py`.
- Connect:
  - Choose `Connection Type` and fill fields:
    - Serial: `COM`, `Baud`.
    - SSH/Telnet: `Host`, `Port`, `Username`, `Password`, optional `Enable password`.
  - Set `Manufacturer` (e.g., `h3c`, `cisco`, `juniper`, `arista`).
  - Click `Connect`.
- Fetch context:
  - `Fetch Running Config` pulls live config for AI context.
  - `Fetch Available Commands` runs `?` and adds output to context and DB.
- Terminal usage:
  - Type commands in the inline input and press `Enter` to send.
  - Use `Send RETURN` to just send Return.
  - Use `?` button to send space + `?` then Return to reveal next tokens.
- Use AI:
  - In Chat, enter a request, select provider/model, and submit.
  - Click per-command `Send to CLI` or use `Push AI Commands to Device` near the terminal.
- Device controls:
  - Click `Enter Privileged/Config Mode` before making changes.
  - Use `Quit` to step back one mode level.
  - Click `Save Config` to persist changes.

AI Providers
- Supported: Gemini, OpenAI, Mistral, Claude, Ollama, Simulation
- Configure in the UI: select provider, enter API key if required, choose model.

API Key Checks
- Use the `Check API Key` button to validate connectivity and credentials for the selected provider.
- Gemini
  - Requires an API key.
  - Sends a minimal request via the Gemini-compatible OpenAI endpoint.
  - Reports success, quota exhaustion, or authorization errors.
- OpenAI
  - Requires an API key.
  - Lists available models to validate the key without needing a specific model.
- Mistral
  - Requires an API key.
  - Performs a lightweight chat request with `mistral-tiny`.
- Claude (Anthropic)
  - Requires an API key.
  - Performs a minimal messages call with `claude-3-opus-20240229`.
- Ollama
  - No API key required.
  - Verifies the local daemon at `http://localhost:11434` and lists models.
- Simulation
  - No API key or connectivity required.

Common errors
- `Unauthorized`: invalid or missing API key for the chosen provider.
- `Quota exceeded` or `Rate limit`: key is valid but usage limits are reached.
- `Host unreachable`: for Ollama, ensure the daemon is running locally.

Command Database
- Build process:
  - Click `Build Command Database` to run two phases:
    - CLI phase: BFS explores command hierarchy with `?`, saving into `command_tree`.
    - AI phase: queries guidance for key topics (VLANs, Interfaces, L3, System, Port-Channel).
- Storage:
  - `cli_cache.db` (SQLite) tables:
    - `available_commands_cache`: raw bulk `?` outputs with context.
    - `command_tree`: unique `manufacturer + command_path` with raw `?` output.
    - `command_knowledge`: AI-generated guidance per category.
- Limits:
  - BFS depth capped (default 4) and branch fan-out limited to avoid explosion.
  - Small delays used to avoid flooding devices.

Notes and Tips
- Set `Manufacturer` correctly; vendor-aware actions and `Netmiko` device type depend on it.
- Pager prompts like `--More--` are auto-handled by sending space during capture.
- At top-level on Cisco, `exit` may log you out; use `Quit` thoughtfully.
- For H3C/Huawei, `Save Config` auto-confirms `Y`.

Dependencies
- `netmiko` for SSH
- `pyserial` for Serial
- `telnetlib` for Telnet (removed in Python 3.13; Telnet may not be available)

Troubleshooting
- Telnet on Python 3.13: `telnetlib` was removed; use Python ≤ 3.12 or prefer SSH.
- Ollama connection issues: ensure `http://localhost:11434` is running and models are pulled.
- Incorrect vendor commands: verify `Manufacturer` is set and provide clear requests.
- Serial COM errors: check permissions and port availability.
- GUI freezes: long operations may block; allow time for fetch/AI steps.

Security
- `profiles.json` stores connection info locally. Handle your machine securely.
- API keys for cloud AI are not persisted across sessions.

Contributing
- Issues and PRs are welcome. Please keep changes focused and consistent with existing style.