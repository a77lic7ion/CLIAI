import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox, ttk, filedialog
import customtkinter as ctk
import json
import os
import re
import requests
import serial
import serial.tools.list_ports
import time
import threading
import queue
import sqlite3
try:
    import telnetlib
except Exception:
    telnetlib = None
try:
    from netmiko import ConnectHandler
except Exception:
    ConnectHandler = None

# --- AI Provider Integration ---
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    genai = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
# Mistral SDK v1: unified client
try:
    from mistralai import Mistral
except ImportError:
    Mistral = None
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    import ollama
except ImportError:
    ollama = None

# --- Netmiko device_type mapping ---
MANUFACTURER_TO_DEVICE_TYPE = {
    "h3c": "hp_comware",
    "cisco": "cisco_ios",
    "juniper": "juniper_junos",
    "arista": "arista_eos",
    "huawei": "huawei",
}

class TelnetAdapter:
    def __init__(self, host: str, port: int):
        if telnetlib is None:
            raise RuntimeError("telnetlib is not available; cannot use Telnet")
        self._tn = telnetlib.Telnet(host, port)
        self._buf = b""

    def write(self, data: bytes):
        if isinstance(data, str):
            data = data.encode("utf-8", errors="ignore")
        self._tn.write(data)

    def _capture(self):
        try:
            chunk = self._tn.read_very_eager()
            if chunk:
                self._buf += chunk
        except Exception:
            pass

    @property
    def in_waiting(self):
        self._capture()
        return len(self._buf)

    def read(self, n: int):
        if not self._buf:
            self._capture()
        data = self._buf[:n]
        self._buf = self._buf[n:]
        return data

    def close(self):
        try:
            self._tn.close()
        except Exception:
            pass

class NetmikoAdapter:
    def __init__(self, **kwargs):
        if ConnectHandler is None:
            raise RuntimeError("netmiko is not available; cannot use SSH")
        self._conn = ConnectHandler(**kwargs)
        self._buf = b""

    def write(self, data: bytes):
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="ignore")
        self._conn.write_channel(data)
        self._capture()

    def _capture(self):
        try:
            s = self._conn.read_channel()
            if s:
                self._buf += s.encode("utf-8", errors="ignore")
        except Exception:
            pass

    @property
    def in_waiting(self):
        self._capture()
        return len(self._buf)

    def read(self, n: int):
        if not self._buf:
            self._capture()
        data = self._buf[:n]
        self._buf = self._buf[n:]
        return data

    def close(self):
        try:
            self._conn.disconnect()
        except Exception:
            pass

class AIProvider:
    def __init__(self):
        self.provider = "None"
        self.api_key = None
        self.client = None

    def set_provider(self, provider, api_key=None):
        self.provider = provider
        self.api_key = api_key
        self.client = None
        try:
            if self.provider == "Gemini" and genai:
                genai.configure(api_key=self.api_key)
            elif self.provider == "OpenAI" and OpenAI:
                self.client = OpenAI(api_key=self.api_key)
            elif self.provider == "Ollama" and ollama:
                self.client = ollama.Client(host='http://localhost:11434' )
            elif self.provider == "Mistral" and Mistral:
                self.client = Mistral(api_key=self.api_key)
            elif self.provider == "Claude" and anthropic:
                self.client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as e:
            messagebox.showerror("AI Initialization Error", f"Failed to initialize {self.provider}: {e}")

    def get_commands(self, user_request, manufacturer, model, version, device_type=None, running_config=None, available_commands=None, use_web_search=False, ollama_model='llama3', gemini_model=None, prompt_style='default'):
        if self.provider == "None": return ["# AI not configured."]
        if self.provider == "Simulation": return self.run_simulation(user_request)

        device_context = f"The target device is a **{manufacturer}**"
        if device_type: device_context += f" of type **{device_type}**"
        if model: device_context += f" model **{model}**"
        if version: device_context += f" running software version **{version}**"
        device_context += "."

        config_context = ""
        if running_config and running_config.strip():
            config_context = f"""**CURRENT DEVICE CONFIGURATION:**
---
{running_config}
---
"""

        available_commands_context = ""
        if available_commands and available_commands.strip():
            available_commands_context = f"""**AVAILABLE COMMANDS IN CURRENT MODE:**
---
{available_commands}
---
"""

        final_context = f"{config_context}{available_commands_context}Based on the context above, and the user's request below, generate the necessary commands."

        # --- A much more forceful and explicit system prompt ---
        if prompt_style == 'guidance':
            system_prompt = f"""
            You are a helpful network agent with 30 years of hands-on experience across Huawei, Juniper, Cisco, and H3C devices. Act as a senior network engineer: explain clearly, troubleshoot configurations, and provide practical, vendor-accurate guidance.

            {final_context}

            **CONTEXT:** The user is working with a **{manufacturer.upper()} {device_type.upper() if device_type else ''}** device. Tailor all guidance to this vendor and device type.

            **YOUR DIRECTIVES:**
            1.  **BE HELPFUL AND EXPLANATORY:** The user is asking a general question. Do not just provide commands. Explain concepts, outline steps, and provide examples.
            2.  **ASK FOR DETAILS (if needed):** If the request is missing critical info (e.g., IPs), state what is needed and why. Use placeholders like `<Your_IP_Address>`.
            3.  **STRUCTURE YOUR RESPONSE:** Use lists/steps and fenced code blocks for command examples.
            4.  **VENDOR ACCURACY:** Ensure examples and guidance align with {manufacturer.upper()} syntax and behaviors.
            """
        elif prompt_style == 'fix_command':
            system_prompt = f"""
            You are a network command correction expert for {manufacturer.upper()} {device_type.upper() if device_type else ''} devices. The user executed a command and received an error. Your task is to provide only the single, corrected, executable CLI command that the user likely intended to run.

            {final_context}

            **USER'S COMMAND:** {user_request.split('ERR_SEPARATOR')[0]}
            **DEVICE ERROR MESSAGE:** {user_request.split('ERR_SEPARATOR')[1]}

            **YOUR DIRECTIVES:**
            1.  **COMMANDS ONLY:** Your entire response must be only the corrected CLI command(s).
            2.  **NO EXPLANATIONS:** Do not add any descriptive text or apologies.
            3.  **NO MARKDOWN:** Do not use markdown code blocks (```).
            4.  **ONE COMMAND PER LINE:** Each command must be on a new line.
            5.  If you cannot determine a correction, return a single line starting with '# AI Error: Unable to determine correction.'
            """
        else: # Default prompt
            system_prompt = f"""
            You are a helpful network agent with 30 years of experience across Huawei, Juniper, Cisco, and H3C. Your task is to generate precise, executable CLI commands tailored to the user's exact platform and context.

            {final_context}

            **CRITICAL CONTEXT:** The target device is a **{manufacturer.upper()} {device_type.upper() if device_type else ''}** device. All commands you generate **MUST** use the correct syntax for this platform.

            **DEVICE DETAILS:** {device_context}

            **YOUR DIRECTIVES:**
            1.  **PRIORITIZE THE PLATFORM:** The user-provided manufacturer/device type ({manufacturer.upper()} {device_type.upper() if device_type else ''}) are primary. Your output must be correct for this platform.
            2.  **COMMANDS ONLY:** Respond only with the CLI commands needed to achieve the user's goal.
            3.  **NO EXPLANATIONS:** Do not add descriptive text, apologies, or intro sentences.
            4.  **NO MARKDOWN:** Do not use markdown code blocks (```).
            5.  **ONE COMMAND PER LINE:** Each command on a new line.
            6.  **HANDLE AMBIGUITY:** If unclear, return a single line starting with '# AI Error:' and a brief reason.
            7.  **ASSUME PRIVILEGED MODE:** Do not include mode-entry commands like 'enable' (Cisco) or 'system-view' (H3C).

            **Example for H3C:**
            User Request: create vlan 100
            Correct Response:
            vlan 100

            **Example for Cisco:**
            User Request: create vlan 100
            Correct Response:
            configure terminal
            vlan 100
            end
            """
        try:
            if self.provider == "Gemini":
                # Use Gemini API via OpenAI-compatible endpoint (chat.completions)
                # Respect explicit selection, fallback to preference order
                model_preferences = [
                    "gemini-1.5-flash",
                    "gemini-2.0-flash",
                    "gemini-1.5-pro"
                ]
                if gemini_model:
                    if not gemini_model.startswith("gemini-"):
                        gemini_model = f"gemini-{gemini_model}"
                    model_preferences = [gemini_model] + [m for m in model_preferences if m != gemini_model]

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_request},
                ]

                try:
                    if OpenAI:
                        client = OpenAI(
                            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                            api_key=self.api_key,
                        )
                        # Try models by preference order
                        last_err = None
                        for m in model_preferences:
                            try:
                                completion = client.chat.completions.create(
                                    model=m,
                                    messages=messages,
                                )
                                content = None
                                try:
                                    content = completion.choices[0].message.content
                                except Exception:
                                    content = completion.choices[0].get("message", {}).get("content", "")
                                return [line for line in (content or "").strip().split("\n") if line]
                            except Exception as err:
                                last_err = err
                                continue
                        raise last_err if last_err else Exception("Gemini OpenAI-compatible call failed")
                    else:
                        # Fallback to direct REST call
                        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
                        last_err_text = None
                        for m in model_preferences:
                            payload = {"model": m, "messages": messages}
                            try:
                                resp = requests.post(f"{url}?key={self.api_key}", json=payload, timeout=30)
                                data = resp.json()
                                if resp.status_code == 200 and "choices" in data:
                                    content = data["choices"][0]["message"]["content"]
                                    return [line for line in content.strip().split("\n")]
                                else:
                                    last_err_text = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
                                    continue
                            except Exception as err:
                                last_err_text = str(err)
                                continue
                        return ["# AI Error: Gemini request failed.", f"# {last_err_text}"]
                except Exception as e:
                    return ["# AI Error: Gemini request failed.", f"# {str(e)}"]

            elif self.provider == "OpenAI":
                if not OpenAI: return ["# OpenAI library not installed."]
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_request}],
                    model="gpt-4o",
                )
                return chat_completion.choices.message.content.strip().split('\n')

            elif self.provider == "Mistral":
                if not Mistral: return ["# Mistral library not installed."]
                if not self.client: return ["# Mistral client not initialized. Set provider again."]
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_request},
                ]
                response = self.client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages,
                )
                try:
                    content = response.choices[0].message.content
                except Exception:
                    content = str(response)
                return [line for line in (content or "").strip().split('\n')]

            elif self.provider == "Claude":
                if not anthropic: return ["# Anthropic library not installed."]
                if not self.client: return ["# Anthropic client not initialized. Set provider again."]
                resp = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=800,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_request}],
                )
                # resp.content is list of blocks; concatenate text
                content = "".join(getattr(block, "text", "") for block in getattr(resp, "content", []) )
                return [line for line in content.strip().split('\n')]

            elif self.provider == "Ollama":
                if not ollama: return ["# Ollama library not installed."]
                if not self.client: return ["# Ollama client not initialized. Set provider again."]
                if not ollama_model or "not found" in ollama_model or "timeout" in ollama_model:
                    return ["# Ollama model not selected or server not available."]
                
                response = self.client.chat(
                    model=ollama_model,
                    messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_request}]
                )
                return response['message']['content'].strip().split('\n')
            
            else:
                return ["# AI not configured or simulation mode."]

        except Exception as e:
            return ["# AI Error: An exception occurred.", f"# {str(e)}"]

    def run_simulation(self, user_request):
        request = user_request.lower()
        if "create vlan 10" in request:
            return ["system-view", "vlan 10", "name Management", "quit"]
        else:
            return ["# AI (Simulated): Request not recognized."]


class NetApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        try:
            ctk.set_appearance_mode("Dark")
            ctk.set_default_color_theme("blue")
        except Exception:
            pass
        self.title("NetIntelli X")
        # Default size aligned with provided screenshot
        self.geometry("1478x768")
        try:
            self.minsize(1200, 700)
        except Exception:
            pass

        self.connection = None
        self.serial_queue = queue.Queue()
        self.reader_thread = None
        self.reader_running = False
        self.profiles = {}
        self.profiles_file = 'profiles.json'
        self.ai_provider = AIProvider()
        # Session memory for auto-corrected commands per context
        self.session_cmd_cache = {}
        self.last_manual_command = None
        self.last_chat_response = None
        # DB build control
        self.db_build_thread = None
        try:
            self.stop_db_build_event = threading.Event()
        except Exception:
            self.stop_db_build_event = None

        # --- Main Content Frame ---
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_pane = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = ctk.CTkFrame(main_pane)
        main_pane.add(left_frame, width=600)

        conn_frame = ctk.CTkFrame(left_frame)
        conn_frame.pack(pady=6, padx=8, fill="x")
        ctk.CTkLabel(conn_frame, text="Connection", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=6, padx=5, pady=(6,4), sticky="w")
        ctk.CTkLabel(conn_frame, text="Profile:").grid(row=1, column=0, padx=5, pady=3, sticky="w")
        self.profile_combo = ctk.CTkComboBox(conn_frame, state="normal")
        self.profile_combo.grid(row=1, column=1, columnspan=2, padx=5, pady=3, sticky="ew")
        try:
            self.profile_combo.configure(command=self.load_selected_profile)
        except Exception:
            pass
        # Connection type selector (Serial/SSH/Telnet)
        ctk.CTkLabel(conn_frame, text="Conn Type:").grid(row=1, column=4, padx=5, pady=3, sticky="w")
        self.conn_type_var = tk.StringVar(value="Serial")
        self.conn_type_combo = ctk.CTkComboBox(conn_frame, values=["Serial", "SSH", "Telnet"], width=120)
        self.conn_type_combo.grid(row=1, column=5, padx=5, pady=3, sticky="w")
        try:
            self.conn_type_combo.configure(command=self.on_conn_type_change)
        except Exception:
            pass
        ctk.CTkLabel(conn_frame, text="COM Port:").grid(row=2, column=0, padx=5, pady=3, sticky="w")
        self.com_port_combo = ctk.CTkComboBox(conn_frame, state="normal")
        self.com_port_combo.grid(row=2, column=1, padx=5, pady=3, sticky="ew")
        ctk.CTkButton(conn_frame, text="Refresh", command=self.refresh_com_ports).grid(row=2, column=2, padx=5, pady=3)
        ctk.CTkLabel(conn_frame, text="Baud:").grid(row=2, column=3, padx=5, pady=3, sticky="w")
        self.baud_combo = ctk.CTkComboBox(conn_frame, values=["9600","19200","38400","57600","115200"], width=100)
        self.baud_combo.grid(row=2, column=4, padx=5, pady=3, sticky="w")
        self.baud_combo.set("9600")
        # Populate COM ports now that the combobox exists
        self.refresh_com_ports()
        ctk.CTkLabel(conn_frame, text="Username:").grid(row=3, column=0, padx=5, pady=3, sticky="w")
        self.user_entry = ctk.CTkEntry(conn_frame)
        self.user_entry.grid(row=3, column=1, padx=5, pady=3, sticky="ew")
        ctk.CTkLabel(conn_frame, text="Password:").grid(row=3, column=2, padx=5, pady=3, sticky="w")
        self.pass_entry = ctk.CTkEntry(conn_frame, show="*")
        self.pass_entry.grid(row=3, column=3, padx=5, pady=3, sticky="ew")
        # Manual login helper: autofill username/password and send to CLI
        self.auto_login_btn = ctk.CTkButton(conn_frame, text="Autofill & Send to CLI", command=self.autofill_send_to_cli)
        self.auto_login_btn.grid(row=3, column=4, padx=5, pady=3, sticky="ew")

        ctk.CTkLabel(conn_frame, text="Enable Pass:").grid(row=4, column=0, padx=5, pady=3, sticky="w")
        self.enable_pass_entry = ctk.CTkEntry(conn_frame, show="*")
        self.enable_pass_entry.grid(row=4, column=1, padx=5, pady=3, sticky="ew")
        # Network connection fields (used for SSH/Telnet)
        ctk.CTkLabel(conn_frame, text="Host:").grid(row=4, column=2, padx=5, pady=3, sticky="w")
        self.host_entry = ctk.CTkEntry(conn_frame)
        self.host_entry.grid(row=4, column=3, padx=5, pady=3, sticky="ew")
        ctk.CTkLabel(conn_frame, text="Port:").grid(row=4, column=4, padx=5, pady=3, sticky="w")
        self.port_entry = ctk.CTkEntry(conn_frame, width=80)
        self.port_entry.grid(row=4, column=5, padx=5, pady=3, sticky="w")

        self.connect_btn = ctk.CTkButton(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=5, column=0, columnspan=2, padx=10, pady=4, sticky="ew")
        # Unified button: Enter privileged/config mode based on manufacturer
        self.enable_btn = ctk.CTkButton(conn_frame, text="Enter Privileged/Config Mode", command=self.enter_privileged_or_config_mode)
        self.enable_btn.grid(row=5, column=2, columnspan=2, padx=10, pady=4, sticky="ew")
        conn_frame.columnconfigure(1, weight=1)

        profile_btn_frame = ctk.CTkFrame(conn_frame)
        profile_btn_frame.grid(row=1, column=2, columnspan=2, sticky='ew')
        ctk.CTkButton(profile_btn_frame, text="Save", command=self.save_profile).pack(side=tk.LEFT, fill='x', expand=True)
        ctk.CTkButton(profile_btn_frame, text="Update", command=self.update_profile).pack(side=tk.LEFT, fill='x', expand=True)
        ctk.CTkButton(profile_btn_frame, text="Delete", command=self.delete_profile).pack(side=tk.LEFT, fill='x', expand=True)

        terminal_frame = ctk.CTkFrame(left_frame)
        terminal_frame.pack(pady=6, padx=8, expand=True, fill="both")
        ctk.CTkLabel(terminal_frame, text="Device Terminal", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(6,2))
        # Use CustomTkinter textbox for better theming and resizing
        self.terminal = ctk.CTkTextbox(terminal_frame)
        try:
            self.terminal.configure(wrap='word', font=("Consolas", 12))
            # Track current terminal font size explicitly for +/- controls
            self.terminal_font_size = 12
        except Exception:
            # Still set a default size tracker even if configure fails
            self.terminal_font_size = getattr(self, 'terminal_font_size', 12)
        self.terminal.pack(expand=True, fill="both")
        # Direct input: type into terminal window and press Enter to send
        self._setup_direct_terminal_input()
        self._show_prompt()

        # Inline terminal input controls
        term_input_frame = ctk.CTkFrame(terminal_frame)
        term_input_frame.pack(fill="x", padx=10, pady=6)
        try:
            self.term_input = ctk.CTkEntry(term_input_frame)
            self.term_input.pack(side=tk.LEFT, fill="x", expand=True)
            # Allow pressing Return to send the typed command from the input field
            try:
                self.term_input.bind("<Return>", self.send_terminal_input)
                # Bind Ctrl+C to send interrupt (ETX)
                self.term_input.bind("<Control-c>", self._on_ctrl_c)
            except Exception:
                pass
            ctk.CTkButton(term_input_frame, text="Send", command=self.send_terminal_input).pack(side=tk.LEFT, padx=6)
            ctk.CTkButton(term_input_frame, text="Send RETURN", command=self.send_enter_key).pack(side=tk.LEFT, padx=6)
            # New help button: sends a space then '?' and presses return
            ctk.CTkButton(term_input_frame, text="?", command=self.send_space_then_question).pack(side=tk.LEFT, padx=6)
            # Ctrl+C button to interrupt long-running device output/commands
            ctk.CTkButton(term_input_frame, text="Ctrl+C", command=self.send_ctrl_c).pack(side=tk.LEFT, padx=6)
            ctk.CTkButton(term_input_frame, text="Quit", command=self.send_quit_command).pack(side=tk.LEFT, padx=6)
        except Exception:
            pass

        # Make Push-to-Device accessible near the terminal as well
        term_actions = ctk.CTkFrame(terminal_frame)
        term_actions.pack(fill="x", padx=10, pady=6)
        ctk.CTkButton(term_actions, text="Push AI Commands to Device", command=self.push_ai_commands).pack(side=tk.LEFT, fill="x", expand=True)
        ctk.CTkButton(term_actions, text="Save Config", command=self.save_device_config).pack(side=tk.LEFT, padx=6)

        # --- Right-hand side layout --- 
        right_master_frame = ctk.CTkFrame(main_pane)
        main_pane.add(right_master_frame)

        # New top frame for side-by-side config sections
        top_right_frame = ctk.CTkFrame(right_master_frame)
        top_right_frame.pack(fill="x", expand=False, pady=4, padx=5)

        # AI Configuration (now on the left of the top-right frame)
        ai_config_frame = ctk.CTkFrame(top_right_frame)
        ai_config_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        ctk.CTkLabel(ai_config_frame, text="AI Configuration", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=2, padx=5, pady=(4,2), sticky="w")
        
        ctk.CTkLabel(ai_config_frame, text="Provider:").grid(row=1, column=0, padx=5, pady=3, sticky="w")
        self.ai_provider_combo = ctk.CTkComboBox(ai_config_frame, values=["None", "Gemini", "OpenAI", "Mistral", "Claude", "Ollama", "Simulation"])
        self.ai_provider_combo.grid(row=1, column=1, padx=5, pady=3, sticky="ew")
        self.ai_provider_combo.set("Gemini")
        try:
            self.ai_provider_combo.configure(command=self.on_ai_provider_change)
        except Exception:
            pass
        ctk.CTkLabel(ai_config_frame, text="API Key:").grid(row=2, column=0, padx=5, pady=3, sticky="w")
        self.api_key_entry = ctk.CTkEntry(ai_config_frame, show="*")
        self.api_key_entry.grid(row=2, column=1, padx=5, pady=3, sticky="ew")
        ctk.CTkLabel(ai_config_frame, text="Ollama Model:").grid(row=3, column=0, padx=5, pady=3, sticky="w")
        self.ollama_model_combo = ctk.CTkComboBox(ai_config_frame, state="disabled")
        self.ollama_model_combo.grid(row=3, column=1, padx=5, pady=3, sticky="ew")
        ctk.CTkLabel(ai_config_frame, text="Gemini Model:").grid(row=4, column=0, padx=5, pady=3, sticky="w")
        self.gemini_model_combo = ctk.CTkComboBox(ai_config_frame, values=["1.5-flash", "2.0-flash", "1.5-pro"])
        self.gemini_model_combo.grid(row=4, column=1, padx=5, pady=3, sticky="ew")
        self.gemini_model_combo.set("2.0-flash")
        # Buttons: Set provider and Check API Key
        self.set_ai_btn = ctk.CTkButton(ai_config_frame, text="Set AI Provider", command=self.set_ai_provider)
        self.set_ai_btn.grid(row=5, column=0, padx=5, pady=3, sticky="ew")
        self.check_api_btn = ctk.CTkButton(ai_config_frame, text="Check API Key", command=self.check_api_key)
        self.check_api_btn.grid(row=5, column=1, padx=5, pady=3, sticky="ew")
        ai_config_frame.columnconfigure(1, weight=1)

        # Terminal Options (moved from left frame to top-right)
        terminal_opts_frame = ctk.CTkFrame(top_right_frame)
        terminal_opts_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
        ctk.CTkLabel(terminal_opts_frame, text="Terminal Options", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(4,2))

        ctk.CTkLabel(terminal_opts_frame, text="Wrap Mode:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.term_wrap_combo = ctk.CTkComboBox(terminal_opts_frame, values=["Wrap (word)", "No wrap"], width=140)
        self.term_wrap_combo.set("Wrap (word)")
        try:
            self.term_wrap_combo.configure(command=self.on_term_wrap_change)
        except Exception:
            pass
        self.term_wrap_combo.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # Compact AI Assistant context next to Terminal Options
        ai_assist_top_frame = ctk.CTkFrame(top_right_frame)
        ai_assist_top_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))
        ctk.CTkLabel(ai_assist_top_frame, text="AI Assistant", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(6,4))
        context_frame = ctk.CTkFrame(ai_assist_top_frame)
        context_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=6, sticky="nsew")
        try:
            for c in range(0, 4):
                context_frame.grid_columnconfigure(c, weight=1, uniform="buttons")
        except Exception:
            pass
        ctk.CTkLabel(context_frame, text="Manufacturer:").grid(row=0, column=0, sticky="w")
        # Use only a dropdown for manufacturer; remove free-text entry
        self.man_combo = ctk.CTkComboBox(context_frame, values=["Cisco", "H3C", "Huawei", "Juniper", "Other"], command=self._on_manufacturer_combo_changed)
        self.man_combo.grid(row=0, column=1, sticky="ew", padx=2)
        # For compatibility, point man_entry to combo (get() works similarly)
        self.man_entry = self.man_combo
        ctk.CTkLabel(context_frame, text="Device Type:").grid(row=1, column=0, sticky="w")
        self.type_entry = ctk.CTkEntry(context_frame)
        self.type_entry.grid(row=1, column=1, sticky="ew", padx=2)
        ctk.CTkLabel(context_frame, text="Model:").grid(row=2, column=0, sticky="w")
        self.model_entry = ctk.CTkEntry(context_frame)
        self.model_entry.grid(row=2, column=1, sticky="ew", padx=2)
        ctk.CTkLabel(context_frame, text="Version:").grid(row=3, column=0, sticky="w")
        self.ver_entry = ctk.CTkEntry(context_frame)
        self.ver_entry.grid(row=3, column=1, sticky="ew", padx=2)

        # Shortened button texts to fit
        self.fetch_info_btn = ctk.CTkButton(context_frame, text="Fetch Info", command=self.fetch_device_info)
        self.fetch_info_btn.grid(row=4, column=0, pady=5, sticky="ew")
        self.build_db_btn = ctk.CTkButton(context_frame, text="Build DB", command=self.build_command_database)
        self.build_db_btn.grid(row=4, column=1, pady=5, sticky="ew")
        self.view_kb_btn = ctk.CTkButton(context_frame, text="View KB", command=self.show_knowledge_base_window)
        self.view_kb_btn.grid(row=4, column=2, pady=5, sticky="ew")
        self.import_json_btn = ctk.CTkButton(context_frame, text="Import CLI", command=self.import_cli_json)
        self.import_json_btn.grid(row=4, column=3, pady=5, sticky="ew")
        context_frame.columnconfigure(1, weight=1)
        context_frame.columnconfigure(2, weight=1)
        font_frame = ctk.CTkFrame(terminal_opts_frame)
        # Place font controls on their own row under the header
        font_frame.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        # Replace segmented control with two simple buttons for reliable repeated clicks
        ctk.CTkButton(font_frame, text="+", width=36, command=self.increase_terminal_font).pack(side=tk.LEFT, padx=2)
        ctk.CTkButton(font_frame, text="-", width=36, command=self.decrease_terminal_font).pack(side=tk.LEFT, padx=2)
        self.auto_pager_var = tk.BooleanVar(value=True)
        # Stack Terminal Options buttons vertically for compact layout
        self.fix_command_btn = ctk.CTkButton(terminal_opts_frame, text="Fix with AI", command=self.ai_fix_last_command, state=tk.DISABLED)
        self.fix_command_btn.grid(row=3, column=0, sticky="ew", padx=5, pady=4)
        ctk.CTkButton(terminal_opts_frame, text="Clear", command=self.clear_terminal).grid(row=4, column=0, sticky="ew", padx=5, pady=4)
        ctk.CTkButton(terminal_opts_frame, text="Export…", command=self.export_terminal_chat).grid(row=5, column=0, sticky="ew", padx=5, pady=4)
        ctk.CTkCheckBox(terminal_opts_frame, text="Auto-pager", variable=self.auto_pager_var).grid(row=6, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkButton(terminal_opts_frame, text="Next Page", command=self.send_pager_next).grid(row=7, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkButton(terminal_opts_frame, text="Stop Paging", command=self.send_pager_stop).grid(row=8, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkButton(terminal_opts_frame, text="Disable Paging", command=self.disable_paging).grid(row=9, column=0, sticky="w", padx=5, pady=2)

        # Apply requested defaults and initialize provider
        try:
            self.api_key_entry.delete(0, tk.END)
            self.api_key_entry.insert(0, "AIzaSyASna_AVZT5NynbpU1eNzXwnxWgo4f_6Lc")
            self.set_ai_provider()
        except Exception:
            pass

        # Bottom: Side-by-side split for AI Assistant and Chat
        right_split = tk.PanedWindow(right_master_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        right_split.pack(fill="both", expand=True)

        ai_assistant_frame = ctk.CTkFrame(right_split)
        right_split.add(ai_assistant_frame, minsize=300)
        # Header label to mimic LabelFrame caption
        ctk.CTkLabel(ai_assistant_frame, text="AI Assistant", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(8,0))

        ttk.Separator(ai_assistant_frame, orient='horizontal').pack(fill='x', pady=5, padx=10)

        # (Fetch Running Config moved to Chat pane)
        # (Moved context panes to Chat window above)
        ttk.Separator(ai_assistant_frame, orient='horizontal').pack(fill='x', pady=5, padx=10)
        ctk.CTkLabel(ai_assistant_frame, text="Your Request:").pack(pady=5, padx=10, anchor="w")
        self.ai_input = ctk.CTkEntry(ai_assistant_frame)
        self.ai_input.pack(pady=5, padx=10, fill="x")
        self.ai_input.bind("<Return>", self.query_ai)
        self.use_web_search_var = tk.BooleanVar(value=True)
        self.web_search_check = ctk.CTkCheckBox(ai_assistant_frame, text="Use Web Search (Gemini)", variable=self.use_web_search_var)
        self.web_search_check.pack(pady=5, padx=10, anchor="w")
        ctk.CTkButton(ai_assistant_frame, text="Generate Commands", command=self.query_ai).pack(pady=5, padx=10, fill="x")
        # Reduce AI output pane height to favor Chat context visibility
        self.ai_output = ctk.CTkTextbox(ai_assistant_frame)
        try:
            self.ai_output.configure(wrap='word', font=("Consolas", 12), height=120)
        except Exception:
            pass
        self.ai_output.pack(pady=10, padx=10, expand=True, fill="both")

        # Chat pane: ask questions, get backend answers, and generate commands for changes
        chat_frame = ctk.CTkFrame(right_split)
        right_split.add(chat_frame, minsize=300)
        ctk.CTkLabel(chat_frame, text="Chat", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(6,2))
        # Context fetchers above the chat window
        chat_context_frame = ctk.CTkFrame(chat_frame)
        chat_context_frame.pack(pady=5, padx=10, fill='x')
        # Button row for fetching and exporting running config
        button_row = ctk.CTkFrame(chat_context_frame)
        button_row.pack(pady=5, fill="x")
        self.fetch_config_btn = ctk.CTkButton(button_row, text="Fetch Running Config for AI Context", command=self.fetch_running_config)
        self.fetch_config_btn.pack(side=tk.LEFT, fill="x", expand=True)
        self.export_config_btn = ctk.CTkButton(button_row, text="Export to TXT", command=self.export_running_config_to_txt)
        self.export_config_btn.pack(side=tk.LEFT, padx=6)
        # Larger running-config viewer for better visibility
        self.running_config_text = ctk.CTkTextbox(chat_context_frame)
        try:
            self.running_config_text.configure(wrap='word', font=("Consolas", 12), height=180)
        except Exception:
            pass
        self.running_config_text.pack(pady=5, expand=True, fill="both")
        self.fetch_q_btn = ctk.CTkButton(chat_context_frame, text="Fetch '?' Commands for AI Context", command=self.fetch_available_commands)
        self.fetch_q_btn.pack(pady=5, fill="x")
        # Option to append fetched '?' output to the CLI command DB
        self.append_available_to_db_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(chat_context_frame, text="Append '?' output to CLI DB", variable=self.append_available_to_db_var).pack(pady=2, anchor='w')
        # Larger '?' commands viewer as well
        self.available_commands_text = ctk.CTkTextbox(chat_context_frame)
        try:
            self.available_commands_text.configure(wrap='word', font=("Consolas", 12), height=150)
        except Exception:
            pass
        self.available_commands_text.pack(pady=5, expand=True, fill="both")

        # Halve the chat agent window height by giving room to context panes
        self.chat_log = ctk.CTkTextbox(chat_frame)
        try:
            self.chat_log.configure(wrap='word', font=("Consolas", 12), height=120)
        except Exception:
            pass
        self.chat_log.pack(pady=6, padx=10, expand=True, fill="both")
        chat_input_frame = ctk.CTkFrame(chat_frame)
        chat_input_frame.pack(fill="x", padx=10, pady=6)
        self.chat_input = ctk.CTkEntry(chat_input_frame)
        self.chat_input.pack(side=tk.LEFT, fill="x", expand=True)
        self.chat_input.bind("<Return>", self.chat_ask)
        ctk.CTkButton(chat_input_frame, text="Send", command=self.chat_ask).pack(side=tk.LEFT, padx=6)
        # Option to append running config to chat queries for exact-device context
        self.append_rc_to_chat_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(chat_input_frame, text="Append Running Config", variable=self.append_rc_to_chat_var).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(chat_input_frame, text="Save to KB", command=self.save_chat_to_knowledge).pack(side=tk.LEFT, padx=6)
        # Sync commands directly from the last chat response to the connected device
        ctk.CTkButton(chat_input_frame, text="Sync Commands", command=self.sync_chat_commands).pack(side=tk.LEFT, padx=6)

        # Favor Chat side for visibility: ~35% AI Assistant, ~65% Chat
        self.after(100, lambda: right_split.sash_place(0, int(right_split.winfo_width() * 0.35), 0))

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        # Replace single label with a frame containing label + progress bar
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar = ctk.CTkLabel(self.status_frame, textvariable=self.status_var)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Bottom progress bar (indeterminate) for visual task progress
        try:
            self.bottom_progress = ttk.Progressbar(self.status_frame, mode="indeterminate", length=140)
            # Hidden by default; shown when tasks run
            self.bottom_progress.pack(side=tk.RIGHT, padx=6, pady=2)
            self.bottom_progress.stop()
            self.bottom_progress.pack_forget()
            self._busy_task_depth = 0
        except Exception:
            self.bottom_progress = None
            self._busy_task_depth = 0
        
        # --- Busy Overlay (progress/thinking indicator) ---
        self._create_busy_overlay()
        
        self.load_profiles()
        self.on_ai_provider_change()
        self.update_status("Ready")
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database to cache AI-generated commands."""
        try:
            self.db_conn = sqlite3.connect('cli_cache.db')
            cursor = self.db_conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generated_commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    manufacturer TEXT NOT NULL,
                    user_request TEXT NOT NULL,
                    generated_commands TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS command_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    manufacturer TEXT NOT NULL,
                    category TEXT NOT NULL,
                    guidance_text TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS command_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    manufacturer TEXT NOT NULL,
                    device_type TEXT,
                    incorrect_command TEXT NOT NULL UNIQUE,
                    corrected_command TEXT NOT NULL
                )
            ''')
            # Cache for available commands ('?' output), appended by user
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS available_commands_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    manufacturer TEXT NOT NULL,
                    context TEXT,
                    commands_text TEXT NOT NULL
                )
            ''')
            # Hierarchical command tree discovered via recursive '?' exploration
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS command_tree (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    manufacturer TEXT NOT NULL,
                    command_path TEXT NOT NULL,
                    context TEXT,
                    raw_output TEXT,
                    UNIQUE(manufacturer, command_path)
                )
            ''')
            self.db_conn.commit()
            self.log_to_terminal("Local command cache database initialized.", "info")
        except Exception as e:
            self.db_conn = None
            self.log_to_terminal(f"Error initializing database: {e}", "error")

    def _save_available_commands_to_db(self, manufacturer, context, commands_text):
        """Append '?' available commands output to local DB for future AI context."""
        if not self.db_conn or not manufacturer or not commands_text:
            return
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "INSERT INTO available_commands_cache (manufacturer, context, commands_text) VALUES (?, ?, ?)",
                (manufacturer, context or '', commands_text)
            )
            self.db_conn.commit()
            self.log_to_terminal("Available commands appended to CLI DB.", "info")
        except Exception as e:
            self.log_to_terminal(f"Failed to save available commands: {e}", "error")

    def _save_command_branch_to_db(self, manufacturer, command_path, context, raw_output, conn=None):
        """Save a discovered command branch (command path and its '?' output) into the hierarchical command tree.
        Uses a provided SQLite connection when running in a worker thread to avoid cross-thread usage issues.
        """
        if not manufacturer or not command_path:
            return

        local_conn = None
        try:
            use_conn = conn if conn is not None else self.db_conn
            if use_conn is None:
                local_conn = sqlite3.connect('cli_cache.db')
                use_conn = local_conn
            cursor = use_conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO command_tree (manufacturer, command_path, context, raw_output) VALUES (?, ?, ?, ?)",
                (manufacturer, command_path, context or '', raw_output or '')
            )
            use_conn.commit()
        except Exception as e:
            self.log_to_terminal(f"Failed to save command branch: {e}", "error")
        finally:
            if local_conn:
                try:
                    local_conn.close()
                except Exception:
                    pass

    def _parse_help_tokens(self, help_text):
        """Parse CLI help ('?') output to extract possible next tokens/keywords.

        Heuristics:
        - Take the first word-like token per line (letters, digits, underscore, hyphen)
        - Ignore placeholders like '<...>' or '[...]'
        - Skip obvious non-command lines (e.g., 'More', banners)
        """
        tokens = []
        if not help_text:
            return tokens
        try:
            for line in help_text.splitlines():
                s = line.strip()
                if not s:
                    continue
                # Skip pager lines and separators
                if re.search(r"-{2,}\s*More|-{2,}", s, re.IGNORECASE):
                    continue
                # Ignore placeholder lines
                if s.startswith('<') or s.startswith('['):
                    continue
                m = re.match(r"^([A-Za-z][A-Za-z0-9_-]*)", s)
                if m:
                    tok = m.group(1)
                    # Filter overly generic or prompt-like artifacts
                    if tok.lower() in {"more", "usage", "help", "system"}:
                        continue
                    if tok not in tokens:
                        tokens.append(tok)
        except Exception:
            pass
        return tokens

    def _save_commands_to_db(self, manufacturer, request, commands):
        """Save a generated command set to the database."""
        if not self.db_conn or not commands or not request or not manufacturer:
            return
        try:
            commands_text = "\n".join(commands)
            cursor = self.db_conn.cursor()
            cursor.execute(
                "INSERT INTO generated_commands (manufacturer, user_request, generated_commands) VALUES (?, ?, ?)",
                (manufacturer, request, commands_text)
            )
            self.db_conn.commit()
        except Exception as e:
            self.log_to_terminal(f"Failed to save commands to local cache: {e}", "error")

    def _save_knowledge_to_db(self, manufacturer, category, guidance):
        """Saves AI-generated guidance to the knowledge base table."""
        if not self.db_conn or not guidance or not category or not manufacturer:
            return
        try:
            guidance_text = "\n".join(guidance)
            cursor = self.db_conn.cursor()
            # Check if an entry for this manufacturer/category already exists
            cursor.execute("SELECT id FROM command_knowledge WHERE manufacturer = ? AND category = ?", (manufacturer, category))
            if cursor.fetchone():
                # Update existing entry
                cursor.execute(
                    "UPDATE command_knowledge SET guidance_text = ?, timestamp = CURRENT_TIMESTAMP WHERE manufacturer = ? AND category = ?",
                    (guidance_text, manufacturer, category)
                )
            else:
                # Insert new entry
                cursor.execute(
                    "INSERT INTO command_knowledge (manufacturer, category, guidance_text) VALUES (?, ?, ?)",
                    (manufacturer, category, guidance_text)
                )
            self.db_conn.commit()
        except Exception as e:
            self.log_to_terminal(f"Failed to save knowledge to local cache: {e}", "error")

    def _save_correction_to_db(self, incorrect_cmd, corrected_cmd):
        """Saves a successful command correction to the database for future use."""
        if not self.db_conn or not incorrect_cmd or not corrected_cmd:
            return
        try:
            manufacturer = self.man_entry.get()
            device_type = self.type_entry.get()
            # Use INSERT OR REPLACE to add the new correction or update an existing one for the same bad command
            cursor = self.db_conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO command_corrections (manufacturer, device_type, incorrect_command, corrected_command) VALUES (?, ?, ?, ?)",
                (manufacturer, device_type, incorrect_cmd, corrected_cmd)
            )
            self.db_conn.commit()
            self.log_to_terminal(f"Saved correction for '{incorrect_cmd}' to local database.", "info")
        except Exception as e:
            self.log_to_terminal(f"Failed to save correction to DB: {e}", "error")

    def build_command_database(self):
        """Starts a background thread to populate the command knowledge base from the AI."""
        manufacturer = self.man_entry.get().strip()
        if not manufacturer:
            messagebox.showerror("Input Error", "Manufacturer must be set to build the command database.")
            return

        if messagebox.askyesno("Confirm", f"This will ask the AI a series of questions to build a knowledge base for '{manufacturer}'. It may take several minutes and incur costs with your AI provider. Continue?"):
            self.log_to_terminal(f"Starting command database build for {manufacturer}...", "info")
            self.update_status(f"Building DB for {manufacturer}...")
            # Prepare stop control and UI state
            try:
                if self.stop_db_build_event:
                    self.stop_db_build_event.clear()
                self._set_build_button_state(True)
            except Exception:
                pass
            # Run the potentially long-running task in a separate thread
            thread = threading.Thread(target=self._build_db_worker, args=(manufacturer,), daemon=True)
            self.db_build_thread = thread
            thread.start()

    def stop_command_database(self):
        """Signal the DB build worker to stop scraping and return control."""
        try:
            if self.stop_db_build_event and not self.stop_db_build_event.is_set():
                self.stop_db_build_event.set()
                self.log_to_terminal("Stop requested: halting command DB build.", "warn")
                self.update_status("Stopping DB build…")
        except Exception:
            pass

    def _build_db_worker(self, manufacturer):
        """The actual worker function to query the AI and populate the DB."""
        db_conn = None
        try:
            db_conn = sqlite3.connect('cli_cache.db')
            cursor = db_conn.cursor()

            topics = {
                "VLANs": "a comprehensive guide to all common commands for creating, configuring, and deleting VLANs, including assigning ports to VLANs in both access and trunk mode.",
                "Interfaces": "a comprehensive guide to all common commands for configuring physical interfaces, including setting speed, duplex, description, and enabling or disabling the interface.",
                "L3 and Routing": "a comprehensive guide to all common commands for Layer 3 configuration, including setting IP addresses on interfaces/VLANs and configuring static routes.",
                "System Management": "a comprehensive guide to all common commands for system management, including setting the hostname, clock, saving the configuration, and viewing system information like version and logs.",
                "Port-Channel": "a comprehensive guide to all common commands for configuring Link Aggregation Groups (LAGs) or Port-Channels."
            }

            # Seed: expand fetched '?' commands using CLI and web search for comprehensive options
            try:
                available_text = self.available_commands_text.get('1.0', tk.END).strip()
            except Exception:
                available_text = ''

            # Fallback: use latest cached '?' output from DB if present
            if not available_text:
                try:
                    cursor.execute(
                        "SELECT commands_text FROM available_commands_cache WHERE manufacturer = ? ORDER BY timestamp DESC LIMIT 1",
                        (manufacturer,)
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        available_text = row[0]
                except Exception:
                    pass

            # If we don't have '?' text yet, try to fetch directly from the device
            if not available_text:
                try:
                    self.log_to_terminal("Fetching '?' help from device for expansion seed...", "info")
                    available_text = self.run_device_command('?', timeout=12)
                except Exception:
                    available_text = ''

            # Determine context from prompt for saving branches
            try:
                last_line = self.terminal.get("end-2l", "end-1l").strip()
                context_guess = ""
                if last_line.startswith('[') and last_line.endswith(']'):
                    context_guess = "System View"
                elif last_line.startswith('<') and last_line.endswith('>'):
                    context_guess = "User View"
                else:
                    context_guess = "Unknown"
            except Exception:
                context_guess = "Unknown"

            # --- Recursively expand commands via CLI '?' and save to DB ---
            def _expand_cli_commands(manuf, seed_text):
                # Discover top-level tokens
                tokens = self._parse_help_tokens(seed_text)
                # Fallback: try 'display ?' or 'show ?' based on platform
                if not tokens:
                    try:
                        if manuf.lower() == 'h3c':
                            t = self.run_device_command('display ?', timeout=6)
                        else:
                            t = self.run_device_command('show ?', timeout=6)
                        tokens = self._parse_help_tokens(t)
                        # Save the top-level branch itself
                        self._save_command_branch_to_db(manuf, 'display' if manuf.lower()=='h3c' else 'show', context_guess, t, conn=db_conn)
                    except Exception:
                        pass

                if not tokens:
                    return

                # Handle early stop
                try:
                    if self.stop_db_build_event and self.stop_db_build_event.is_set():
                        return
                except Exception:
                    pass
                # Sort tokens alphabetically to start from 'A'
                try:
                    tokens = sorted(set(tokens), key=lambda x: x.lower())
                except Exception:
                    pass
                visited = set()
                max_depth = 4
                queue = []
                for tok in tokens:
                    path = [tok]
                    queue.append(path)

                while queue:
                    # Check stop signal before exploring next branch
                    try:
                        if self.stop_db_build_event and self.stop_db_build_event.is_set():
                            self.log_to_terminal("Stopping CLI '?' expansion…", "info")
                            break
                    except Exception:
                        pass
                    path = queue.pop(0)
                    cmd_path = ' '.join(path)
                    # Avoid duplicate exploration
                    if cmd_path in visited:
                        continue
                    visited.add(cmd_path)

                    # Query '?' for the current path
                    try:
                        # Ensure we are at the main starting view before each new exploration
                        try:
                            self._ensure_root_prompt(manuf)
                        except Exception:
                            pass
                        # Stop check before executing device command
                        try:
                            if self.stop_db_build_event and self.stop_db_build_event.is_set():
                                break
                        except Exception:
                            pass
                        help_out = self.run_device_command(f"{cmd_path} ?", timeout=8)
                    except Exception:
                        help_out = ''

                    # Persist branch with raw help output
                    try:
                        self._save_command_branch_to_db(manuf, cmd_path, context_guess, help_out, conn=db_conn)
                    except Exception:
                        pass

                    # If the device responded with an error, skip expanding this branch
                    if self._is_cli_error(help_out):
                        # Attempt to return to root view in case the device changed mode unexpectedly
                        try:
                            self._ensure_root_prompt(manuf)
                        except Exception:
                            pass
                        continue

                    # If depth limit reached, don't expand further
                    if len(path) >= max_depth:
                        continue

                    # Parse next tokens from this help
                    next_tokens = self._parse_help_tokens(help_out)
                    # Limit branching to avoid explosion
                    next_tokens = next_tokens[:50]
                    for nt in next_tokens:
                        new_path = path + [nt]
                        new_cmd = ' '.join(new_path)
                        if new_cmd not in visited:
                            queue.append(new_path)
                    # Small delay to avoid flooding device
                    time.sleep(0.2)

            try:
                self.log_to_terminal("Starting CLI '?' expansion to build command tree...", "info")
                self.update_status("Building DB: Expanding CLI commands...")
                # Stop check before starting expansion
                try:
                    if self.stop_db_build_event and self.stop_db_build_event.is_set():
                        raise Exception("Build stopped before CLI expansion")
                except Exception:
                    pass
                _expand_cli_commands(manufacturer, available_text)
                self.log_to_terminal("CLI command tree expansion complete.", "info")
            except Exception as e:
                self.log_to_terminal(f"CLI expansion failed: {e}", "error")

            for category, question in topics.items():
                # Stop check between categories
                try:
                    if self.stop_db_build_event and self.stop_db_build_event.is_set():
                        self.update_status("Command DB build stopped.")
                        self.log_to_terminal("DB build stopped before category queries.", "warn")
                        break
                except Exception:
                    pass
                full_request = f"For a {manufacturer} device, provide {question}"
                self.log_to_terminal(f"Querying AI for category: {category}...", "info")
                self.update_status(f"Building DB: Querying for {category}...")

                guidance = self.ai_provider.get_commands(
                    full_request,
                    manufacturer,
                    self.model_entry.get(),
                    self.ver_entry.get(),
                    device_type=self.type_entry.get(),
                    running_config=self.running_config_text.get('1.0', tk.END) if hasattr(self, 'running_config_text') else None,
                    available_commands=available_text,
                    use_web_search=True,
                    prompt_style='guidance'
                )

                if guidance and not guidance[0].strip().startswith("# AI Error:"):
                    guidance_text = "\n".join(guidance)
                    cursor.execute("SELECT id FROM command_knowledge WHERE manufacturer = ? AND category = ?", (manufacturer, category))
                    if cursor.fetchone():
                        cursor.execute(
                            "UPDATE command_knowledge SET guidance_text = ?, timestamp = CURRENT_TIMESTAMP WHERE manufacturer = ? AND category = ?",
                            (guidance_text, manufacturer, category)
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO command_knowledge (manufacturer, category, guidance_text) VALUES (?, ?, ?)",
                            (manufacturer, category, guidance_text)
                        )
                    self.log_to_terminal(f"Successfully retrieved knowledge for {category}.", "info")
                else:
                    self.log_to_terminal(f"Failed to get guidance for {category}. Response: {guidance[0] if guidance else 'No response'}", "error")
                # Respect stop requests without lingering sleep
                try:
                    if self.stop_db_build_event and self.stop_db_build_event.is_set():
                        break
                except Exception:
                    pass
                time.sleep(5) # Add a small delay to avoid hitting API rate limits

            # If we have '?' commands fetched, ask AI to expand each with syntax/parameters/examples
            if available_text:
                # Stop check before expansion request
                try:
                    if self.stop_db_build_event and self.stop_db_build_event.is_set():
                        self.update_status("Command DB build stopped.")
                        self.log_to_terminal("DB build stopped before '?' expansion.", "warn")
                        raise Exception("Build stopped before '?' expansion")
                except Exception:
                    pass
                self.log_to_terminal("Querying AI to expand '?' command list using web search...", "info")
                self.update_status("Building DB: Expanding '?' commands...")
                expand_request = (
                    "Using the following available commands list captured from the device ('?'), "
                    "expand each command with full syntax, parameters/options, common variations, and short examples. "
                    "Group related commands and include cross-references."
                )
                guidance = self.ai_provider.get_commands(
                    expand_request,
                    manufacturer,
                    self.model_entry.get(),
                    self.ver_entry.get(),
                    device_type=self.type_entry.get(),
                    running_config=self.running_config_text.get('1.0', tk.END) if hasattr(self, 'running_config_text') else None,
                    available_commands=available_text,
                    use_web_search=True,
                    prompt_style='guidance'
                )
                if guidance and not guidance[0].strip().startswith("# AI Error:"):
                    guidance_text = "\n".join(guidance)
                    cursor.execute(
                        "SELECT id FROM command_knowledge WHERE manufacturer = ? AND category = ?",
                        (manufacturer, "Available Commands Expanded")
                    )
                    if cursor.fetchone():
                        cursor.execute(
                            "UPDATE command_knowledge SET guidance_text = ?, timestamp = CURRENT_TIMESTAMP WHERE manufacturer = ? AND category = ?",
                            (guidance_text, manufacturer, "Available Commands Expanded")
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO command_knowledge (manufacturer, category, guidance_text) VALUES (?, ?, ?)",
                            (manufacturer, "Available Commands Expanded", guidance_text)
                        )
                    self.log_to_terminal("Successfully expanded '?' commands and saved to KB.", "info")
                else:
                    self.log_to_terminal(
                        f"Failed to expand '?' commands. Response: {guidance[0] if guidance else 'No response'}",
                        "error"
                    )

            # Append a build summary to Knowledge Base
            try:
                model_val = self.model_entry.get().strip()
                version_val = self.ver_entry.get().strip()
                profile_val = self.profile_combo.get().strip() if hasattr(self, 'profile_combo') else ''
                summary_text = (
                    "Build Summary\n"
                    f"Manufacturer: {manufacturer}\n"
                    f"Model: {model_val}\n"
                    f"Version: {version_val}\n"
                    f"Profile: {profile_val}\n"
                    "Categories: VLANs, Interfaces, L3 and Routing, System Management, Port-Channel\n"
                ) + ("Included '?' command expansion\n" if available_text else "") + "Status: Success"
                cursor.execute(
                    "INSERT INTO command_knowledge (manufacturer, category, guidance_text) VALUES (?, ?, ?)",
                    (manufacturer, "Build Summary", summary_text)
                )
            except Exception as e:
                self.log_to_terminal(f"Failed to append build summary to KB: {e}", "error")

            db_conn.commit()
            # Final status depending on stop state
            try:
                stopped = bool(self.stop_db_build_event and self.stop_db_build_event.is_set())
            except Exception:
                stopped = False
            if stopped:
                self.log_to_terminal("Command database build stopped by user.", "warn")
                self.update_status("Command DB build stopped.")
            else:
                self.log_to_terminal("Command database build finished and saved.", "info")
                self.update_status("Command DB build finished.")
        except Exception as e:
            self.log_to_terminal(f"Database build worker failed: {e}", "error")
            self.update_status("Command DB build failed.")
        finally:
            if db_conn:
                db_conn.close()
            # Restore UI buttons
            try:
                self._set_build_button_state(False)
            except Exception:
                pass

    def _set_build_button_state(self, building: bool):
        """Toggle the Build Cmd DB button into a Stop button while building."""
        try:
            if building:
                self.build_db_btn.configure(text="Stop Build", command=self.stop_command_database, state=tk.NORMAL)
            else:
                self.build_db_btn.configure(text="Build Cmd DB", command=self.build_command_database, state=tk.NORMAL)
        except Exception:
            pass

    def import_cli_json(self):
        """Import CLI-related JSON and append data into cli_cache.db tables.
        Supports keys: command_corrections, command_knowledge, generated_commands.
        If a list is provided at top-level, it is treated as corrections.
        """
        try:
            filepath = filedialog.askopenfilename(title="Select CLI JSON", filetypes=[("JSON files", "*.json")])
            if not filepath:
                return

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not self.db_conn:
                self._init_db()

            cursor = self.db_conn.cursor()
            imported_corrections = 0
            imported_kb = 0
            imported_generated = 0

            # Normalize structures
            if isinstance(data, dict):
                corrections = data.get('command_corrections') or data.get('corrections') or []
                knowledge = data.get('command_knowledge') or data.get('knowledge') or []
                generated = data.get('generated_commands') or data.get('generated') or []
            elif isinstance(data, list):
                corrections = data
                knowledge = []
                generated = []
            else:
                messagebox.showerror("Import Error", "Unsupported JSON structure.")
                return

            # Corrections
            for item in corrections or []:
                if not isinstance(item, dict):
                    continue
                manufacturer = (item.get('manufacturer') or self.man_entry.get() or '').strip()
                device_type = (item.get('device_type') or self.type_entry.get() or '').strip()
                incorrect = item.get('incorrect_command') or item.get('incorrect') or item.get('bad')
                corrected = item.get('corrected_command') or item.get('correct') or item.get('fix')
                if manufacturer and incorrect and corrected:
                    cursor.execute(
                        "INSERT OR REPLACE INTO command_corrections (manufacturer, device_type, incorrect_command, corrected_command) VALUES (?, ?, ?, ?)",
                        (manufacturer, device_type, incorrect, corrected)
                    )
                    imported_corrections += 1

            # Knowledge entries
            for item in knowledge or []:
                if not isinstance(item, dict):
                    continue
                manufacturer = (item.get('manufacturer') or self.man_entry.get() or '').strip()
                category = (item.get('category') or '').strip()
                guidance = item.get('guidance_text') or item.get('guidance') or item.get('text') or ''
                guidance_text = "\n".join(guidance) if isinstance(guidance, list) else str(guidance)
                if manufacturer and category and guidance_text:
                    cursor.execute("SELECT id FROM command_knowledge WHERE manufacturer = ? AND category = ?", (manufacturer, category))
                    if cursor.fetchone():
                        cursor.execute(
                            "UPDATE command_knowledge SET guidance_text = ?, timestamp = CURRENT_TIMESTAMP WHERE manufacturer = ? AND category = ?",
                            (guidance_text, manufacturer, category)
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO command_knowledge (manufacturer, category, guidance_text) VALUES (?, ?, ?)",
                            (manufacturer, category, guidance_text)
                        )
                    imported_kb += 1

            # Generated command sets
            for item in generated or []:
                if not isinstance(item, dict):
                    continue
                manufacturer = (item.get('manufacturer') or self.man_entry.get() or '').strip()
                request = item.get('user_request') or item.get('request') or ''
                cmds = item.get('generated_commands') or item.get('commands') or ''
                cmds_text = "\n".join(cmds) if isinstance(cmds, list) else str(cmds)
                if manufacturer and request and cmds_text:
                    cursor.execute(
                        "INSERT INTO generated_commands (manufacturer, user_request, generated_commands) VALUES (?, ?, ?)",
                        (manufacturer, request, cmds_text)
                    )
                    imported_generated += 1

            self.db_conn.commit()
            summary = f"Imported {imported_corrections} corrections, {imported_kb} knowledge entries, {imported_generated} generated sets from {os.path.basename(filepath)}"
            self.log_to_terminal(summary, "info")
            try:
                messagebox.showinfo("Import Complete", summary)
            except Exception:
                pass
        except Exception as e:
            try:
                messagebox.showerror("Import Error", f"Failed to import JSON: {e}")
            except Exception:
                pass
            self.log_to_terminal(f"Failed to import JSON: {e}", "error")

    def update_status(self, message):
        self.status_var.set(message)
        self.update_idletasks()

    def _create_busy_overlay(self):
        try:
            self.busy_var = tk.StringVar(value="Working…")
            self.busy_frame = tk.Frame(self, bg="#000000", highlightthickness=0)
            # Inner panel
            inner = tk.Frame(self.busy_frame, bg="#222222", bd=2, relief=tk.RIDGE)
            inner.place(relx=0.5, rely=0.5, anchor="center")
            tk.Label(inner, textvariable=self.busy_var, fg="white", bg="#222222", font=("Arial", 10, "bold")).pack(padx=20, pady=(20, 10))
            self.busy_bar = ttk.Progressbar(inner, mode="indeterminate", length=220)
            self.busy_bar.pack(padx=20, pady=(0, 20))
            # Hidden by default
            self.busy_frame.place_forget()
        except Exception:
            # If overlay cannot be created, fail silently
            self.busy_frame = None
            self.busy_bar = None

    def set_busy(self, is_busy=True, message=None):
        """Show/Hide an overlay with an indeterminate progress bar.
        Also updates the status message if provided.
        """
        try:
            # Maintain task depth to support nested operations
            if is_busy:
                self._busy_task_depth = max(0, getattr(self, '_busy_task_depth', 0)) + 1
            else:
                self._busy_task_depth = max(0, getattr(self, '_busy_task_depth', 0) - 1)

            if message:
                self.busy_var.set(message)
                self.update_status(message)
            if is_busy:
                if self.busy_frame:
                    self.busy_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
                    self.busy_frame.lift()
                if self.busy_bar:
                    self.busy_bar.start(10)
                # Bottom progress bar (always show while any tasks are active)
                if self.bottom_progress:
                    try:
                        # If currently hidden, re-pack it to show
                        if not str(self.bottom_progress).endswith(".n"):  # dummy check to avoid errors
                            pass
                        # Ensure it’s packed (idempotent: packing the same widget again is safe)
                        self.bottom_progress.pack(side=tk.RIGHT, padx=6, pady=2)
                        self.bottom_progress.start(10)
                    except Exception:
                        pass
            else:
                # Only stop/hide when no nested tasks remain
                if self._busy_task_depth == 0:
                    if self.busy_bar:
                        self.busy_bar.stop()
                    if self.busy_frame:
                        self.busy_frame.place_forget()
                    if self.bottom_progress:
                        try:
                            self.bottom_progress.stop()
                            self.bottom_progress.pack_forget()
                        except Exception:
                            pass
            self.update_idletasks()
        except Exception:
            pass

    def on_ai_provider_change(self, event=None):
        provider = self.ai_provider_combo.get()
        self.api_key_entry.configure(state=tk.NORMAL if provider in ["Gemini", "OpenAI", "Mistral", "Claude"] else tk.DISABLED)
        self.web_search_check.configure(state=tk.NORMAL if provider == "Gemini" else tk.DISABLED)
        # Enable Gemini model selection only for Gemini provider
        try:
            self.gemini_model_combo.configure(state="readonly" if provider == "Gemini" else tk.DISABLED)
        except Exception:
            pass
        
        if provider == "Ollama":
            self.ollama_model_combo.configure(state="readonly")
            self.fetch_ollama_models()
        else:
            self.ollama_model_combo.configure(state=tk.DISABLED)
        try:
            self.ollama_model_combo.configure(values=[])
        except Exception:
            pass
        self.ollama_model_combo.set('')

    def fetch_ollama_models(self):
        self.log_to_terminal("Fetching Ollama models...", "info")
        self.set_busy(True, "Fetching Ollama models…")
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            response.raise_for_status()
            models = [m.get('name') for m in response.json().get('models', []) if m.get('name')]
            if models:
                try:
                    self.ollama_model_combo.configure(values=models)
                except Exception:
                    pass
                self.ollama_model_combo.set(models[0])
                self.log_to_terminal(f"Found {len(models)} Ollama models.", "info")
                self.update_status("Ollama models loaded.")
            else:
                try:
                    self.ollama_model_combo.configure(values=["No models found"])
                except Exception:
                    pass
                self.ollama_model_combo.set("No models found")
                self.update_status("Ollama: No models found.")
        except requests.exceptions.Timeout:
            self.log_to_terminal("Ollama server timed out. Is it running and responsive?", "error")
            self.update_status("Ollama server timed out.")
            try:
                self.ollama_model_combo.configure(values=["Ollama server timeout"])
            except Exception:
                pass
            self.ollama_model_combo.set("Ollama server timeout")
        except requests.exceptions.ConnectionError:
            self.log_to_terminal("Ollama server not found at http://localhost:11434.", "error")
            self.update_status("Ollama server not found.")
            try:
                self.ollama_model_combo.configure(values=["Ollama server not found"])
            except Exception:
                pass
            self.ollama_model_combo.set("Ollama server not found")
        except Exception as e:
            self.log_to_terminal(f"Error fetching Ollama models: {e}", "error")
            self.update_status("Error fetching Ollama models.")
            try:
                self.ollama_model_combo.configure(values=["Error fetching models"])
            except Exception:
                pass
            self.ollama_model_combo.set("Error fetching models")
        finally:
            self.set_busy(False)

    def refresh_com_ports(self):
        """Refresh the list of available COM ports"""
        try:
            # Guard against early calls before UI is built
            if not hasattr(self, 'com_port_combo'):
                return
            ports = serial.tools.list_ports.comports()
            available_ports = [port.device for port in ports if port.device]
            try:
                self.com_port_combo.configure(values=available_ports)
            except Exception:
                pass
            if available_ports and not self.com_port_combo.get():
                self.com_port_combo.set(available_ports[0])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh COM ports: {str(e)}")

    def on_conn_type_change(self, event=None):
        """Set sensible defaults for port based on connection type."""
        try:
            ctype = (self.conn_type_var.get() or 'Serial').strip()
            if ctype == 'SSH':
                self.port_entry.delete(0, tk.END)
                self.port_entry.insert(0, '22')
            elif ctype == 'Telnet':
                self.port_entry.delete(0, tk.END)
                self.port_entry.insert(0, '23')
            else:
                # Serial: clear host/port to avoid confusion
                self.port_entry.delete(0, tk.END)
        except Exception:
            pass

    def get_selected_gemini_model_full(self):
        sel = getattr(self, 'gemini_model_combo', None)
        if not sel:
            return None
        val = self.gemini_model_combo.get().strip()
        if not val:
            return None
        return val if val.startswith("gemini-") else f"gemini-{val}"

    def check_api_key(self):
        provider = self.ai_provider_combo.get()
        api_key = self.api_key_entry.get().strip()
        # Providers requiring a key
        providers_require_key = {"Gemini", "OpenAI", "Mistral", "Claude"}
        if provider in providers_require_key and not api_key:
            messagebox.showerror("Error", "API Key is required to check.")
            return
        if provider == "Gemini":
            model = self.get_selected_gemini_model_full() or "gemini-1.5-flash"
            messages = [{"role": "user", "content": "ping"}]
            try:
                if OpenAI:
                    client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key)
                    client.chat.completions.create(model=model, messages=messages)
                    messagebox.showinfo("API Key", "Gemini API key is valid and reachable.")
                else:
                    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
                    resp = requests.post(f"{url}?key={api_key}", json={"model": model, "messages": messages, "max_tokens": 1}, timeout=20)
                    if resp.status_code == 200:
                        messagebox.showinfo("API Key", "Gemini API key is valid and reachable.")
                    elif resp.status_code == 429:
                        messagebox.showinfo("API Key", "Gemini API key is valid, but quota is exhausted.")
                    elif resp.status_code == 401:
                        messagebox.showerror("API Key", "Unauthorized: API key invalid or not permitted.")
                    else:
                        messagebox.showerror("API Key", f"Error: HTTP {resp.status_code} - {resp.text}")
            except Exception as e:
                messagebox.showerror("API Key", f"Check failed: {e}")
        elif provider == "OpenAI":
            try:
                if not OpenAI:
                    raise RuntimeError("OpenAI SDK not installed")
                client = OpenAI(api_key=api_key)
                # Listing models avoids dependency on a specific model name
                _ = client.models.list()
                messagebox.showinfo("API Key", "OpenAI API key is valid and reachable.")
            except Exception as e:
                messagebox.showerror("API Key", f"OpenAI check failed: {e}")
        elif provider == "Mistral":
            try:
                if Mistral:
                    client = Mistral(api_key=api_key)
                    # Use a tiny model for a quick ping
                    client.chat.complete(model="mistral-tiny", messages=[{"role": "user", "content": "ping"}])
                    messagebox.showinfo("API Key", "Mistral API key is valid and reachable.")
                else:
                    # Fallback to REST: list models to validate key without SDK
                    resp = requests.get(
                        "https://api.mistral.ai/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=20,
                    )
                    if resp.status_code == 200:
                        messagebox.showinfo("API Key", "Mistral API key is valid and reachable.")
                    elif resp.status_code == 429:
                        messagebox.showinfo("API Key", "Mistral API key is valid, but quota is exhausted.")
                    elif resp.status_code == 401:
                        messagebox.showerror("API Key", "Unauthorized: API key invalid or not permitted.")
                    else:
                        messagebox.showerror("API Key", f"Error: HTTP {resp.status_code} - {resp.text}")
            except Exception as e:
                messagebox.showerror("API Key", f"Mistral check failed: {e}")
        elif provider == "Claude":
            try:
                if not anthropic:
                    raise RuntimeError("Anthropic SDK not installed")
                client = anthropic.Anthropic(api_key=api_key)
                client.messages.create(model="claude-3-opus-20240229", max_tokens=16, messages=[{"role": "user", "content": "ping"}])
                messagebox.showinfo("API Key", "Claude API key is valid and reachable.")
            except Exception as e:
                messagebox.showerror("API Key", f"Claude check failed: {e}")
        elif provider == "Ollama":
            try:
                if not ollama:
                    raise RuntimeError("Ollama SDK not installed")
                client = ollama.Client(host='http://localhost:11434')
                # Ensure Ollama daemon reachable and models listable
                _ = client.list()
                messagebox.showinfo("Ollama", "Ollama is reachable at http://localhost:11434.")
            except Exception as e:
                messagebox.showerror("Ollama", f"Ollama host check failed: {e}")
        else:
            messagebox.showinfo("API Key", "No API key required for this provider.")

    def toggle_connection(self):
        if self.connection:
            self.disconnect()
        else:
            self.connect()

    def on_term_wrap_change(self, event=None):
        mode = self.term_wrap_combo.get()
        try:
            wrap_mode = 'none' if mode == "No wrap" else 'word'
            self.terminal.configure(wrap=wrap_mode)
        except Exception:
            pass

    def _on_manufacturer_selected(self, event=None):
        try:
            sel = self.man_combo.get().strip()
            # Normalize to lowercase for internal handling
            normalized = sel.lower()
            # Auto-populate Device Type based on mapping; blank for 'other'
            try:
                dev_type = MANUFACTURER_TO_DEVICE_TYPE.get(normalized, "")
                self.type_entry.delete(0, tk.END)
                if dev_type:
                    self.type_entry.insert(0, dev_type)
            except Exception:
                pass
        except Exception:
            pass

    def _on_manufacturer_combo_changed(self, choice: str):
        """Wrapper to support CTkComboBox 'command' callback.
        Keeps existing behavior by delegating to _on_manufacturer_selected.
        """
        try:
            # Ensure combo reflects the selected value, then reuse existing handler
            if choice:
                self.man_combo.set(choice)
        except Exception:
            pass
        self._on_manufacturer_selected(None)

    def clear_terminal(self):
        try:
            self.terminal.delete('1.0', tk.END)
        except Exception:
            pass
        try:
            self._show_prompt()
        except Exception:
            pass

    def run_device_command(self, cmd, timeout=4):
        """Send a command to the connected device and capture its response."""
        if not self.connection or not getattr(self, 'is_connected', False):
            raise RuntimeError("Not connected to any device.")

        self._pause_serial_reader()
        try:
            self.log_to_terminal(f"\n> {cmd}", "command")
            self.connection.write((cmd + '\r\n').encode())
            
            response = ""
            start_time = time.time()
            pager_re = re.compile(r"\s*-{2,}\s*More\s*-{2,}\s*|--More--")

            while time.time() - start_time < timeout:
                if self.connection.in_waiting > 0:
                    try:
                        chunk = self.connection.read(self.connection.in_waiting).decode('utf-8', errors='ignore')
                        response += chunk

                        # If we see a pager prompt, send a space and reset the timer
                        if pager_re.search(response):
                            self.connection.write(b' ')
                            # Clean the pager text from our response buffer to avoid re-matching
                            response = pager_re.sub('', response)
                            start_time = time.time() # Reset timeout

                    except Exception as e:
                        self.log_to_terminal(f"[Read Error] {e}", "error")
                        break # Exit loop on read error
                else:
                    time.sleep(0.2) # Wait a bit longer for the next chunk of data
            
            # Clean up any lingering backspaces or control characters from the final output
            response = re.sub(r'.\x08', '', response)

            if response:
                self.log_to_terminal(response, "output")
            else:
                self.log_to_terminal("(no response)", "info")
            return response
        finally:
            self._resume_serial_reader()

    def _is_cli_error(self, output):
        """Detect common CLI error responses across vendors."""
        if not output:
            return True
        text = output.lower()
        patterns = [
            r"unrecognized command",
            r"unknown command",
            r"invalid input",
            r"incomplete command",
            r"too many parameters",
            r"ambiguous command",
            r"syntax error",
            r"% ?error",
            r"% ?invalid",
        ]
        try:
            for p in patterns:
                if re.search(p, text):
                    return True
        except Exception:
            pass
        return False

    def _ensure_root_prompt(self, manufacturer):
        """Ensure the device is at the main starting view before sending expansion commands.

        For H3C/Huawei VRP, remain in System View and avoid quitting out.
        If currently in User View ('<...>'), enter System View with 'system-view'.
        Do not treat hyphens in hostnames as feature-view indicators.
        For Cisco/Arista IOS-like prompts, send 'end' and 'exit' as needed to leave (config*) modes.
        For Juniper, detect '[edit …]' and use 'top' then 'exit' to leave configuration.
        For other vendors, only wake the line to avoid accidental logout.
        """
        try:
            manuf = (manufacturer or '').lower()
            if not self.connection or not getattr(self, 'is_connected', False):
                return

            # Inspect the last prompt line
            try:
                last_line = self.terminal.get("end-2l", "end-1l").strip()
            except Exception:
                last_line = ''

            # H3C/Huawei VRP: ensure System View without quitting out
            if ('h3c' in manuf or 'huawei' in manuf):
                # If we are in user view (<...>), enter system view.
                if last_line.startswith('<') and last_line.endswith('>'):
                    try:
                        self.connection.write(b"system-view\r\n")
                    except Exception:
                        pass
                    # brief settle
                    time.sleep(0.2)
                else:
                    # Already in a bracketed prompt (System/feature view). Do not send 'quit'. Wake line only.
                    try:
                        self.connection.write(b"\r\n")
                    except Exception:
                        pass
            elif ('cisco' in manuf or 'arista' in manuf):
                # Cisco/Arista: leave any (config*) submode; prefer 'end' to jump to exec
                def in_config_prompt(s: str):
                    return '(config' in s

                # If clearly in config, send 'end' once
                if in_config_prompt(last_line):
                    try:
                        self.connection.write(b"end\r\n")
                    except Exception:
                        pass
                    time.sleep(0.2)
                    # Read a short burst to update terminal
                    t0 = time.time()
                    while time.time() - t0 < 0.5:
                        try:
                            waiting = self.connection.in_waiting
                        except Exception:
                            waiting = 0
                        if waiting:
                            try:
                                chunk = self.connection.read(waiting).decode('utf-8', errors='ignore')
                                self.log_to_terminal(chunk, 'output')
                            except Exception:
                                pass
                        time.sleep(0.08)
                    try:
                        last_line = self.terminal.get("end-2l", "end-1l").strip()
                    except Exception:
                        last_line = ''
                    # If still stuck in a deeper config, use 'exit' a few times
                    tries = 0
                    while in_config_prompt(last_line) and tries < 3:
                        try:
                            self.connection.write(b"exit\r\n")
                        except Exception:
                            break
                        time.sleep(0.2)
                        tries += 1
                        try:
                            last_line = self.terminal.get("end-2l", "end-1l").strip()
                        except Exception:
                            last_line = ''
                else:
                    # Not obviously in config; just wake the line
                    try:
                        self.connection.write(b"\r\n")
                    except Exception:
                        pass
            elif ('juniper' in manuf):
                # Juniper: detect '[edit ...]' in recent lines; then 'top' and 'exit'
                try:
                    recent = self.terminal.get('end-8l', 'end-1l')
                except Exception:
                    recent = ''
                def in_edit_mode(text: str):
                    return '[edit' in text
                if in_edit_mode(recent):
                    try:
                        self.connection.write(b"top\r\n")
                        time.sleep(0.1)
                        self.connection.write(b"exit\r\n")
                    except Exception:
                        pass
                else:
                    # Wake line only
                    try:
                        self.connection.write(b"\r\n")
                    except Exception:
                        pass
            else:
                # Unknown/other vendors: wake the line only
                try:
                    self.connection.write(b"\r\n")
                except Exception:
                    pass
        except Exception:
            # Best-effort; ignore failures
            pass

    def _execute_vlan_show_with_autocorrect(self, manufacturer):
        """Run VLAN show command; if it fails, probe help and correct automatically. Returns (output, cmd_used)."""
        m = (manufacturer or '').strip().lower()
        cache_key = (m, 'vlan_show')
        cached = self.session_cmd_cache.get(cache_key)
        if cached:
            out = self.run_device_command(cached, timeout=6)
            if not self._is_cli_error(out):
                return out, cached
            # Cached failed; invalidate
            self.session_cmd_cache.pop(cache_key, None)

        base = self._get_vlan_show_command(m)
        out = self.run_device_command(base, timeout=6)
        if not self._is_cli_error(out):
            self.session_cmd_cache[cache_key] = base
            return out, base

        # Autocorrect by vendor
        try:
            if m == 'h3c':
                h1 = self.run_device_command('display ?', timeout=3)
                cmd = 'display vlan'
                if 'vlan' not in (h1 or '').lower():
                    cmd = 'display vlan'  # still try standard
                h2 = self.run_device_command('display vlan ?', timeout=3)
                # Prefer minimal form unless options are required
                cmd_try = cmd
                out2 = self.run_device_command(cmd_try, timeout=6)
                if self._is_cli_error(out2):
                    # If 'all' appears as option, try it
                    if 'all' in (h2 or '').lower():
                        cmd_try = 'display vlan all'
                        out2 = self.run_device_command(cmd_try, timeout=6)
                if not self._is_cli_error(out2):
                    self.session_cmd_cache[cache_key] = cmd_try
                    return out2, cmd_try

            elif m == 'cisco':
                h1 = self.run_device_command('show ?', timeout=3)
                cmd = 'show vlan'
                if 'vlan' not in (h1 or '').lower():
                    cmd = 'show vlan'
                h2 = self.run_device_command('show vlan ?', timeout=3)
                cmd_try = 'show vlan brief' if 'brief' in (h2 or '').lower() else cmd
                out2 = self.run_device_command(cmd_try, timeout=6)
                if self._is_cli_error(out2):
                    # fallback plain
                    cmd_try = 'show vlan'
                    out2 = self.run_device_command(cmd_try, timeout=6)
                if not self._is_cli_error(out2):
                    self.session_cmd_cache[cache_key] = cmd_try
                    return out2, cmd_try

            elif m == 'arista':
                h2 = self.run_device_command('show vlan ?', timeout=3)
                cmd_try = 'show vlan brief' if 'brief' in (h2 or '').lower() else 'show vlan'
                out2 = self.run_device_command(cmd_try, timeout=6)
                if not self._is_cli_error(out2):
                    self.session_cmd_cache[cache_key] = cmd_try
                    return out2, cmd_try

            elif m == 'juniper':
                cmd_try = 'show vlans'
                out2 = self.run_device_command(cmd_try, timeout=6)
                if not self._is_cli_error(out2):
                    self.session_cmd_cache[cache_key] = cmd_try
                    return out2, cmd_try
        except Exception:
            pass

        # If all else fails, return first attempt
        return out, base

    def increase_terminal_font(self):
        # Increase font size up to a sensible maximum
        try:
            current = getattr(self, 'terminal_font_size', 12)
            new_size = min(current + 1, 48)
            self.terminal_font_size = new_size
            self.terminal.configure(font=("Consolas", new_size))
            # Apply same size to AI, Fetch '?', and Generate Commands panes
            try:
                if hasattr(self, 'ai_output') and self.ai_output:
                    self.ai_output.configure(font=("Consolas", new_size))
            except Exception:
                pass
            try:
                if hasattr(self, 'available_commands_text') and self.available_commands_text:
                    self.available_commands_text.configure(font=("Consolas", new_size))
            except Exception:
                pass
            try:
                if hasattr(self, 'chat_log') and self.chat_log:
                    self.chat_log.configure(font=("Consolas", new_size))
            except Exception:
                pass
            try:
                if hasattr(self, 'running_config_text') and self.running_config_text:
                    self.running_config_text.configure(font=("Consolas", new_size))
            except Exception:
                pass
        except Exception:
            pass

    def decrease_terminal_font(self):
        # Decrease font size down to a sensible minimum
        try:
            current = getattr(self, 'terminal_font_size', 12)
            new_size = max(current - 1, 6)
            self.terminal_font_size = new_size
            self.terminal.configure(font=("Consolas", new_size))
            # Apply same size to AI, Fetch '?', and Generate Commands panes
            try:
                if hasattr(self, 'ai_output') and self.ai_output:
                    self.ai_output.configure(font=("Consolas", new_size))
            except Exception:
                pass
            try:
                if hasattr(self, 'available_commands_text') and self.available_commands_text:
                    self.available_commands_text.configure(font=("Consolas", new_size))
            except Exception:
                pass
            try:
                if hasattr(self, 'chat_log') and self.chat_log:
                    self.chat_log.configure(font=("Consolas", new_size))
            except Exception:
                pass
            try:
                if hasattr(self, 'running_config_text') and self.running_config_text:
                    self.running_config_text.configure(font=("Consolas", new_size))
            except Exception:
                pass
        except Exception:
            pass

    def _on_font_segment(self, value: str):
        try:
            if value == "+":
                self.increase_terminal_font()
            elif value == "-":
                self.decrease_terminal_font()
        except Exception:
            pass

    def export_terminal_chat(self):
        # Export terminal contents to a .txt file via save dialog
        try:
            content = self.terminal.get('1.0', tk.END).strip()
            if not content:
                messagebox.showinfo("Export", "Terminal is empty.")
                return
            path = filedialog.asksaveasfilename(
                title="Export Terminal Chat",
                defaultextension=".txt",
                filetypes=[["Text Files", "*.txt"], ["All Files", "*.*"]]
            )
            if not path:
                return
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content + "\n")
            messagebox.showinfo("Export", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    def _get_vlan_show_command(self, manufacturer):
        m = (manufacturer or '').strip().lower()
        if m == 'h3c':
            return 'display vlan'
        if m == 'cisco':
            return 'show vlan brief'
        if m == 'juniper':
            return 'show vlans'
        if m == 'arista':
            return 'show vlan'
        return 'show vlan'

    def _parse_vlans(self, manufacturer, output):
        """Parse VLANs from device output. Returns list of dicts {id, name}."""
        vlans = []
        m = (manufacturer or '').strip().lower()
        # Clean common control chars (e.g., backspace 0x08) and split lines
        safe_output = (output or '').replace('\x08', '')
        lines = safe_output.splitlines()
        # Common patterns
        cisco_rx = re.compile(r"^(\d+)\s+([\w\-]+)", re.IGNORECASE)
        h3c_id_rx = re.compile(r"VLAN\s*ID\s*:\s*(\d+)", re.IGNORECASE)
        h3c_name_rx = re.compile(r"Name\s*:\s*([^\n]+)", re.IGNORECASE)
        juniper_rx = re.compile(r"^VLAN\s*:\s*([\w\-]+)\s*(\d+)", re.IGNORECASE)
        arista_rx = re.compile(r"^(\d+)\s+([\w\-]+)", re.IGNORECASE)

        if m == 'cisco' or m == 'arista':
            for line in lines:
                mm = cisco_rx.search(line) if m == 'cisco' else arista_rx.search(line)
                if mm:
                    vid = mm.group(1)
                    name = mm.group(2)
                    # Skip header rows like 'VLAN Name'
                    if vid.lower() == 'vlan' or name.lower() == 'name':
                        continue
                    vlans.append({'id': vid, 'name': name})
        elif m == 'h3c':
            # Pattern A: per-VLAN detail blocks
            current = {}
            for line in lines:
                mid = h3c_id_rx.search(line)
                if mid:
                    if current:
                        vlans.append(current)
                    current = {'id': mid.group(1), 'name': ''}
                    continue
                mname = h3c_name_rx.search(line)
                if mname:
                    nm = mname.group(1).strip()
                    if current:
                        current['name'] = nm
            if current:
                vlans.append(current)

            # Pattern B: summary list like "The VLANs include: 1(default), 10, 20, ..."
            try:
                # Find line that mentions include
                include_idx = None
                for i, line in enumerate(lines):
                    if 'vlans include' in line.lower():
                        include_idx = i
                        break
                if include_idx is not None:
                    # Accumulate items from the include line and a few lines after
                    list_text = ''
                    # grab remainder of the include line after ':' if present
                    part = lines[include_idx]
                    mm = re.search(r"include\s*:([^\n]*)", part, re.IGNORECASE)
                    if mm:
                        list_text += mm.group(1)
                    # Append following lines until prompt or blank
                    for j in range(include_idx + 1, min(include_idx + 6, len(lines))):
                        nxt = lines[j].strip()
                        if not nxt:
                            break
                        if nxt.startswith('[') or nxt.lower().startswith('total vlan') or nxt.lower().startswith('display '):
                            break
                        list_text += ' ' + nxt
                    # Parse comma-separated items
                    tokens = [t.strip() for t in list_text.split(',') if t.strip()]
                    for tok in tokens:
                        mm2 = re.search(r"^(\d{1,4})(?:\(([^)]+)\))?", tok)
                        if mm2:
                            vid = mm2.group(1)
                            nm = mm2.group(2) or ''
                            vlans.append({'id': vid, 'name': nm})
            except Exception:
                pass
        elif m == 'juniper':
            for line in lines:
                mm = juniper_rx.search(line)
                if mm:
                    name = mm.group(1)
                    vid = mm.group(2)
                    vlans.append({'id': vid, 'name': name})
        else:
            # Fallback: find any lines starting with a VLAN number
            generic_rx = re.compile(r"^(\d{1,4})\s+([\w\-]*)", re.IGNORECASE)
            for line in lines:
                mm = generic_rx.search(line)
                if mm:
                    vlans.append({'id': mm.group(1), 'name': mm.group(2)})
        # Deduplicate by id
        seen = set()
        uniq = []
        for v in vlans:
            if v['id'] not in seen:
                uniq.append(v)
                seen.add(v['id'])
        return uniq

    def _summarize_vlans(self, vlans):
        if not vlans:
            return "No VLANs found or unable to parse."
        parts = []
        for v in vlans:
            name = v.get('name') or '-'
            parts.append(f"{v['id']}({name})")
        return f"Found {len(vlans)} VLANs: " + ", ".join(parts)

    def send_pager_next(self):
        # Manually advance pager one page (space)
        try:
            if self.connection and getattr(self, "is_connected", False):
                self.connection.write(b' ')
        except Exception:
            pass

    def send_pager_stop(self):
        # Stop paging (q)
        try:
            if self.connection and getattr(self, "is_connected", False):
                self.connection.write(b'q')
        except Exception:
            pass

    def send_enter_key(self):
        """Sends a single ENTER (CR+LF) to the device."""
        try:
            if self.connection and getattr(self, "is_connected", False):
                self.connection.write(b'\r\n')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send ENTER: {e}")

    def send_space_then_question(self):
        """Sends a space, then '?' and presses RETURN to request CLI help."""
        try:
            if not self.connection or not getattr(self, "is_connected", False):
                messagebox.showerror("Error", "Not connected to any device.")
                return
            # Echo to terminal for user feedback
            self.log_to_terminal("\n>  ?", "command")
            # Send a space, then '?' followed by CRLF
            self.connection.write(b' ')
            time.sleep(0.02)
            self.connection.write(b'?\r\n')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send '?': {e}")

    def send_ctrl_c(self):
        """Send Ctrl+C (ETX, 0x03) to interrupt current device operation."""
        try:
            if not self.connection or not getattr(self, "is_connected", False):
                messagebox.showerror("Error", "Not connected to any device.")
                return
            # Echo to terminal for user feedback
            self.log_to_terminal("\n^C", "command")
            # Transmit ETX; works for Serial, Telnet, and Netmiko channel
            self.connection.write(b"\x03")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send Ctrl+C: {e}")

    def _on_ctrl_c(self, event=None):
        try:
            self.send_ctrl_c()
        except Exception:
            pass
        return "break"

    def send_quit_command(self):
        """Go back one mode step using vendor-aware command.

        - H3C/Huawei: send 'quit' (VRP system-view/feature view back step)
        - Cisco/Juniper/others: send 'exit'
        """
        try:
            if not self.connection or not getattr(self, "is_connected", False):
                messagebox.showerror("Error", "Not connected to any device.")
                return

            manufacturer = (self.man_entry.get() or '').strip().lower()
            cmd = 'quit' if ('h3c' in manufacturer or 'huawei' in manufacturer) else 'exit'
            # Reuse core send path for consistency (echo + CRLF + prompt refresh)
            self._send_command(cmd)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send quit/exit: {e}")

    def autofill_send_to_cli(self):
        """Send username and password to the device, each followed by RETURN.

        This provides a manual login option when auto-login did not succeed.
        """
        try:
            if not self.connection or not getattr(self, 'is_connected', False):
                messagebox.showerror("Error", "Not connected to any device.")
                return

            username = ''
            password = ''
            try:
                username = (self.user_entry.get() or '').strip()
            except Exception:
                username = ''
            try:
                password = (self.pass_entry.get() or '')
            except Exception:
                password = ''

            if not username and not password:
                messagebox.showinfo("Missing Credentials", "Please enter a username and/or password.")
                return

            # Informational log without exposing the password
            self.log_to_terminal("[manual-login] Sending username and password to device...", "info")

            # Send username then password, each with CRLF
            try:
                if username:
                    self.connection.write((username + '\r\n').encode())
                else:
                    # If username is empty, at least send a RETURN to advance
                    self.connection.write(b'\r\n')
            except Exception as e:
                self.log_to_terminal(f"[Send error] Username send failed: {e}", "error")

            # Wait for device to present Password prompt (some devices are slow)
            try:
                time.sleep(3.0)
            except Exception:
                pass

            try:
                if password:
                    self.connection.write((password + '\r\n').encode())
                else:
                    # Send RETURN even if no password to proceed on devices without passwords
                    self.connection.write(b'\r\n')
            except Exception as e:
                self.log_to_terminal(f"[Send error] Password send failed: {e}", "error")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to send credentials: {e}")

    def disable_paging(self):
        """Disable terminal paging for the current vendor/session."""
        if not self.connection or not getattr(self, 'is_connected', False):
            messagebox.showerror("Error", "Not connected to any device.")
            return
        vendor = (self.man_entry.get() or '').strip().lower()
        # Choose vendor-specific commands to disable paging
        commands = []
        if vendor.startswith('h3c') or vendor == 'hp' or vendor == 'hp_comware':
            commands = ['screen-length disable']
        elif vendor.startswith('cisco') or vendor == 'ios':
            commands = ['terminal length 0']
        elif vendor.startswith('arista'):
            commands = ['terminal length 0']
        elif vendor.startswith('juniper'):
            commands = ['set cli screen-length 0']
        else:
            # Safe default commonly supported
            commands = ['terminal length 0']

        self._pause_serial_reader()
        try:
            for c in commands:
                self.log_to_terminal(f"\n> {c}", "command")
                self.connection.write((c + '\r\n').encode())
                out = ""
                start = time.time()
                pager_re = re.compile(r"\s*-{2,}\s*More\s*-{2,}\s*|--More--|----\s*More\s*----", re.IGNORECASE)
                while time.time() - start < 3:
                    if self.connection.in_waiting > 0:
                        chunk = self.connection.read(self.connection.in_waiting).decode('utf-8', errors='ignore')
                        out += chunk
                        if pager_re.search(out):
                            self.connection.write(b' ')
                            out = pager_re.sub('', out)
                            start = time.time()
                    time.sleep(0.1)
                if out:
                    self.log_to_terminal(out, 'output')
        finally:
            self._resume_serial_reader()

    def connect(self):
        ctype = (self.conn_type_var.get() or 'Serial').strip()
        username = self.user_entry.get()
        password = self.pass_entry.get()

        if ctype == 'Serial':
            com_port = self.com_port_combo.get()
            try:
                baudrate = int(self.baud_combo.get()) if hasattr(self, 'baud_combo') and self.baud_combo.get() else 9600
            except Exception:
                baudrate = 9600
            if not com_port:
                messagebox.showerror("Error", "Please select a COM port")
                return
            self.set_busy(True, f"Connecting to {com_port}…")
            try:
                # Create serial connection
                self.connection = serial.Serial(
                    port=com_port,
                    baudrate=baudrate,
                    timeout=0,
                    write_timeout=3
                )
                self.connection.reset_input_buffer()
                self.connection.reset_output_buffer()
                self.is_connected = True
                self.connect_btn.configure(text="Disconnect")
                self.log_to_terminal(f"Connected to {com_port}")
                self.update_status(f"Connected to {com_port}")
                self.run_precheck()
                self._resume_serial_reader()
            except Exception as e:
                messagebox.showerror("Connection Error", f"Failed to connect to {com_port}: {str(e)}")
            finally:
                self.set_busy(False)
            return

        # Network connection path
        host = (self.host_entry.get() or '').strip()
        port_val = (self.port_entry.get() or '').strip()
        port = None
        try:
            port = int(port_val) if port_val else (22 if ctype == 'SSH' else 23)
        except Exception:
            port = 22 if ctype == 'SSH' else 23
        if not host:
            messagebox.showerror("Error", "Please enter a host for network connection")
            return

        self.set_busy(True, f"Connecting to {ctype} {host}:{port}…")
        try:
            if ctype == 'Telnet':
                self.connection = TelnetAdapter(host, port)
            else:
                manufacturer = (self.man_entry.get() or '').strip().lower()
                device_type = MANUFACTURER_TO_DEVICE_TYPE.get(manufacturer, 'cisco_ios')
                nm_kwargs = {
                    'device_type': device_type,
                    'host': host,
                    'username': username,
                    'password': password,
                    'port': port,
                    'fast_cli': False,
                }
                self.connection = NetmikoAdapter(**nm_kwargs)

            self.is_connected = True
            self.connect_btn.configure(text="Disconnect")
            self.log_to_terminal(f"Connected to {ctype} {host}:{port}")
            self.update_status(f"Connected to {ctype} {host}:{port}")
            self.run_precheck()
            self._resume_serial_reader()
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect via {ctype}: {str(e)}")
        finally:
            self.set_busy(False)

    def disconnect(self):
        if self.connection:
            self.set_busy(True, "Disconnecting…")
            try:
                self._pause_serial_reader()
                self.connection.close()
                self.connection = None
                self.is_connected = False
                self.connect_btn.configure(text="Connect")
                self.log_to_terminal("Disconnected")
                self.update_status("Disconnected")
            except Exception as e:
                messagebox.showerror("Disconnection Error", f"Error during disconnection: {str(e)}")
            finally:
                self.set_busy(False)

    def _resume_serial_reader(self):
        if self.reader_thread and self.reader_thread.is_alive():
            return
        self.reader_running = True
        self.reader_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
        self.reader_thread.start()
        # Start draining queue on the Tk mainloop
        self.after(50, self._drain_serial_queue)

    def _pause_serial_reader(self):
        self.reader_running = False
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0) # Wait for thread to stop

    def _serial_reader_loop(self):
        while self.reader_running and self.connection:
            try:
                # Read any available bytes
                waiting = self.connection.in_waiting
                if waiting:
                    data = self.connection.read(waiting)
                    # Add a small delay and try a second read to 'stick' reads together.
                    # This helps prevent pager prompts from being split across two reads.
                    time.sleep(0.05)
                    if self.connection.in_waiting > 0:
                        data += self.connection.read(self.connection.in_waiting)

                    if data:
                        try:
                            text = data.decode('utf-8', errors='ignore')
                        except Exception:
                            text = data.decode('latin-1', errors='ignore')
                        # Auto-advance device pager prompts if enabled
                        try:
                            if getattr(self, 'auto_pager_var', None) and self.auto_pager_var.get():
                                if re.search(r"-{2,}\s*More\s*-{2,}|--More--|----\s*More\s*----", text, re.IGNORECASE):
                                    # Send a single space to advance one page
                                    self.connection.write(b' ')
                        except Exception:
                            pass
                        self.serial_queue.put(text)
                else:
                    # Small sleep to avoid busy loop
                    time.sleep(0.05)
            except Exception as e:
                # Push error info and stop reader
                self.serial_queue.put(f"\n[Serial error] {e}\n")
                break

    def _drain_serial_queue(self):
        drained = False
        try:
            while True:
                text = self.serial_queue.get_nowait()
                drained = True
                self.log_to_terminal(text, "output")
                # Check for errors to enable the fix button
                if self._is_cli_error(text):
                    self.last_error_output = text
                    self.fix_command_btn.configure(state=tk.NORMAL)

        except queue.Empty:
            pass
        if self.reader_running:
            # Continue polling while reader active
            self.after(50 if drained else 100, self._drain_serial_queue)

    # --- Direct typing support in terminal ---
    def _setup_direct_terminal_input(self):
        try:
            # Always type at the end of the terminal
            self.terminal.bind("<Key>", self._terminal_force_end)
            # Handle BackSpace within current input line only
            self.terminal.bind("<BackSpace>", self._terminal_on_backspace)
            # Handle Enter to send command
            self.terminal.bind("<Return>", self._terminal_on_return)
        except Exception:
            pass

    def _show_prompt(self):
        try:
            # Set a mark where logs should insert (before the prompt)
            self.terminal.mark_set('PROMPT', tk.END)
            self.terminal.mark_gravity('PROMPT', tk.RIGHT)
            # Insert prompt
            self.terminal.insert(tk.END, "\n> ", "prompt")
            self.terminal.see(tk.END)
            # Mark start of user input after prompt
            self.terminal.mark_set('INPUT_START', tk.END)
            self.terminal.mark_gravity('INPUT_START', tk.LEFT)
        except Exception:
            pass

    def _terminal_force_end(self, event):
        # Keep insertion point at end so typing doesn't edit prior logs
        try:
            if event.keysym == "Return":
                return None
            self.terminal.mark_set(tk.INSERT, tk.END)
        except Exception:
            pass
        return None

    def _terminal_on_backspace(self, event):
        try:
            # Constrain deletion to current input line
            self.terminal.mark_set(tk.INSERT, tk.END)
            try:
                start = self.terminal.index('INPUT_START')
            except Exception:
                start = None
            if not start:
                return None
            if self.terminal.compare(tk.END, "==", start):
                return "break"  # nothing to delete
            # Delete single char before end
            self.terminal.delete("%s-1c" % tk.END, tk.END)
            return "break"
        except Exception:
            return None

    def _terminal_on_return(self, event):
        try:
            try:
                start = self.terminal.index('INPUT_START')
            except Exception:
                start = None
            if not start:
                text = ''
            else:
                text = self.terminal.get(start, tk.END).strip()
            # Clear the current input line before sending
            if start:
                self.terminal.delete(start, tk.END)
            self._send_command(text)
            return "break"
        except Exception:
            return "break"

    def send_terminal_input(self, event=None):
        # Support both legacy Entry and direct terminal typing
        cmd = None
        if hasattr(self, 'term_input') and self.term_input:
            cmd = self.term_input.get().strip()
            self.term_input.delete(0, tk.END)
        else:
            try:
                start = self.terminal.index('INPUT_START')
                cmd = self.terminal.get(start, tk.END).strip()
                # Clear the input area
                self.terminal.delete(start, tk.END)
            except Exception:
                cmd = ''
        self._send_command(cmd)

    def _send_command(self, cmd):
        # Reset the fixer state each time a new command is sent
        self.fix_command_btn.configure(state=tk.DISABLED)
        self.last_manual_command = None
        self.last_error_output = None

        # Local CLI: clear terminal regardless of connection
        if cmd.lower() in ("clear", "cls"):
            self.clear_terminal()
            return
        if not cmd:
            # If no text, treat as plain ENTER
            try:
                self.send_enter_key()
            except Exception:
                pass
            # Refresh prompt after plain enter
            try:
                self._show_prompt()
            except Exception:
                pass
            return
        if not self.connection or not getattr(self, "is_connected", False):
            messagebox.showerror("Error", "Not connected to any device")
            return
        try:
            # Echo command right at the end so it’s visible near the prompt
            self._echo_command(cmd)
            # Send to device
            self.connection.write((cmd + "\r\n").encode())
            self.last_manual_command = cmd # Save for the fixer
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send: {e}")
            self.log_to_terminal(f"\n[Send error] {e}\n", "error")
        finally:
            try:
                self._show_prompt()
            except Exception:
                pass

    def _echo_command(self, cmd):
        # Echo the command near the prompt so the user sees what was sent
        try:
            self.terminal.tag_config("command", foreground="yellow")
            self.terminal.insert(tk.END, f"\n> {cmd}\n", "command")
            self.terminal.see(tk.END)
        except Exception:
            pass

    def _get_ai_correction(self, failed_cmd, error_msg):
        """Gets a command correction using a multi-step process: local DB, AI, then guidance."""
        manufacturer = self.man_entry.get()
        device_type = self.type_entry.get()
        if not manufacturer:
            return ["# AI Error: Manufacturer not set."]

        # 1. Check local corrections database first
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT corrected_command FROM command_corrections WHERE incorrect_command = ? AND manufacturer = ?", (failed_cmd, manufacturer))
            result = cursor.fetchone()
            if result:
                self.log_to_terminal(f"Found correction for '{failed_cmd}' in local database.", "info")
                return result[0].split('\n')
        except Exception as e:
            self.log_to_terminal(f"Failed to query local corrections DB: {e}", "error")

        # 2. If not in local DB, try AI
        self.set_busy(True, f"Asking AI for correction for: {failed_cmd}...")
        request_with_error = f"{failed_cmd}ERR_SEPARATOR{error_msg or 'Unknown error'}"
        corrected_commands = self.ai_provider.get_commands(
            request_with_error, manufacturer, self.model_entry.get(), self.ver_entry.get(),
            device_type=device_type, running_config=self.running_config_text.get('1.0', tk.END),
            prompt_style='fix_command'
        )
        self.set_busy(False)

        # If AI can't find a specific fix, ask for general guidance instead
        if corrected_commands and corrected_commands[0].strip().startswith("# AI Error: Unable to determine correction"):
            self.set_busy(True, "No specific fix found, asking for guidance…")
            guidance_request = f"The command '{failed_cmd}' was incomplete or ambiguous. What are the possible valid commands that could follow it on a {manufacturer} {device_type} device? Provide a brief guide with examples."
            guidance = self.ai_provider.get_commands(
                guidance_request, manufacturer, self.model_entry.get(), self.ver_entry.get(),
                device_type=device_type, running_config=self.running_config_text.get('1.0', tk.END),
                prompt_style='guidance'
            )
            self.set_busy(False)
            return guidance
        
        return corrected_commands

    def ai_fix_last_command(self):
        if not self.last_manual_command:
            messagebox.showerror("Error", "No previous command to fix.")
            return

        self.update_status("Asking AI to fix the last command...")
        
        correction_or_guidance = self._get_ai_correction(self.last_manual_command, self.last_error_output)

        is_fix = not (correction_or_guidance and correction_or_guidance[0].strip().startswith("#"))

        self.ai_output.delete('1.0', tk.END)
        self.ai_output.insert(tk.END, f"> Original command: {self.last_manual_command}\r\n")
        if is_fix:
            self.ai_output.insert(tk.END, "AI Corrected Command:\r\n")
            self.update_status("AI correction received.")
        else:
            self.ai_output.insert(tk.END, "AI Guidance:\r\n")
            self.update_status("AI guidance received.")
        
        self.ai_output.insert(tk.END, "------------------------\r\n")
        self.ai_output.insert(tk.END, "\n".join(correction_or_guidance))
        self.fix_command_btn.configure(state=tk.DISABLED)

    def run_precheck(self):
        """Run precheck for serial connection and auto-login using GUI credentials.

        - Wakes the line with ENTER.
        - Detects common login prompts (Login/Username/User name).
        - Sends username and password automatically if provided.
        - Avoids echoing the password to the terminal.
        """
        self.log_to_terminal("\nRunning pre-check to identify device...", "info")
        self.set_busy(True, "Running pre-check…")
        self.update_idletasks()

        username = (getattr(self, 'user_entry', None).get() if hasattr(self, 'user_entry') else '').strip()
        password = getattr(self, 'pass_entry', None).get() if hasattr(self, 'pass_entry') else ''

        # Helper regexes for prompts
        login_re = re.compile(r"(^|\n)\s*(login|username|user name|user)\s*:", re.IGNORECASE)
        pass_re = re.compile(r"(^|\n)\s*password\s*:", re.IGNORECASE)
        press_enter_re = re.compile(r"press\s+(enter|return)\s+to\s+get\s+started", re.IGNORECASE)

        sent_user = False
        sent_pass = False

        try:
            # Wake the line
            try:
                self.connection.write(b"\r\n")
            except Exception:
                pass

            start = time.time()
            buffer = ""
            # Read/respond loop (pre-reader-thread)
            while time.time() - start < 8:
                waiting = 0
                try:
                    waiting = self.connection.in_waiting
                except Exception:
                    waiting = 0
                if waiting:
                    chunk = self.connection.read(waiting).decode('utf-8', errors='ignore')
                    buffer += chunk
                    # Show device output
                    self.log_to_terminal(chunk, "output")

                    lower = buffer.lower()
                    # Some devices ask to press ENTER first
                    if press_enter_re.search(lower):
                        try:
                            self.connection.write(b"\r\n")
                        except Exception:
                            pass

                    # Send username when prompted
                    if not sent_user and username and login_re.search(lower):
                        try:
                            self.connection.write((username + "\r\n").encode())
                            self.log_to_terminal("[auto-login] Sent username.", "info")
                            sent_user = True
                        except Exception:
                            pass

                    # Send password when prompted
                    if not sent_pass and password and pass_re.search(lower):
                        try:
                            self.connection.write((password + "\r\n").encode())
                            self.log_to_terminal("[auto-login] Sent password (hidden).", "info")
                            sent_pass = True
                        except Exception:
                            pass

                    # If we appear to have a prompt and we already sent credentials, exit precheck
                    if sent_pass or (sent_user and not password):
                        if re.search(r"(^|\n).*[>#]\s*$", buffer):
                            break
                else:
                    # Small wait and try to wake the line once more
                    time.sleep(0.2)
                    try:
                        self.connection.write(b"\r\n")
                    except Exception:
                        pass

            # If no output captured, note it for user
            if not buffer:
                self.log_to_terminal("No initial response from device", "info")

        except Exception as e:
            self.log_to_terminal(f"Precheck error: {str(e)}", "error")
        finally:
            self.set_busy(False)

    def query_ai(self, event=None):
        user_request = self.ai_input.get()
        if not user_request: return
        
        manufacturer = self.man_entry.get()
        if not manufacturer:
            messagebox.showerror("Input Error", "Manufacturer field is required for AI query.")
            return

        self.ai_output.delete('1.0', tk.END)
        self.ai_output.insert(tk.END, f"> User: {user_request}\n\n")
        self.set_busy(True, "Querying AI…")
        self.update_idletasks()
        
        commands = self.ai_provider.get_commands(
            user_request,
            manufacturer,
            self.model_entry.get(),
            self.ver_entry.get(),
            device_type=self.type_entry.get(),
            running_config=self.running_config_text.get('1.0', tk.END),
            available_commands=self.available_commands_text.get('1.0', tk.END),
            use_web_search=self.use_web_search_var.get(),
            ollama_model=self.ollama_model_combo.get(),
            gemini_model=self.get_selected_gemini_model_full()
        )
        self.set_busy(False)
        
        self.ai_output.insert(tk.END, "AI Generated Commands:\n")
        self.ai_output.insert(tk.END, "------------------------\n")
        self.ai_output.insert(tk.END, "\n".join(commands))
        self.update_status("AI response received.")

        # Save successful, non-error commands to the database
        if commands and not any("# AI Error:" in cmd for cmd in commands):
            self._save_commands_to_db(manufacturer, user_request, commands)

    def chat_ask(self, event=None):
        """Chat: answer device info questions by querying backend; generate commands for changes."""
        text = self.chat_input.get().strip()
        if not text:
            return
        self.chat_log.insert(tk.END, f"You: {text}\n")
        # If requested, append current running config to the query for exact-device context
        try:
            if getattr(self, 'append_rc_to_chat_var', None) and self.append_rc_to_chat_var.get():
                rc_text = (self.running_config_text.get('1.0', tk.END) if hasattr(self, 'running_config_text') else '').strip()
                if rc_text:
                    # Limit size to keep prompt manageable
                    rc_text = rc_text[:50000]
                    text = f"{text}\n\nRUNNING_CONFIG_CONTEXT_BEGIN\n{rc_text}\nRUNNING_CONFIG_CONTEXT_END"
                    self.chat_log.insert(tk.END, "(Context: running config appended to query)\n")
        except Exception:
            pass
        self.chat_input.delete(0, tk.END)

        manufacturer = (self.man_entry.get() or '').strip()
        if not manufacturer:
            self.chat_log.insert(tk.END, "System: Please set Manufacturer in AI Configuration.\n\n")
            self.update_status("Chat needs manufacturer context")
            return

        lower = text.lower()
        change_words = ["add", "create", "remove", "delete", "modify", "change", "rename", "assign", "set"]
        mentions_vlan = "vlan" in lower or "vlans" in lower
        is_change = any(w in lower for w in change_words)

        # --- Search local knowledge base first ---
        kb_result = self._search_knowledge_base(text)
        if kb_result:
            guidance, category = kb_result
            self.chat_log.insert(tk.END, f"AI: [From Local KB - '{category}']\n")
            self.chat_log.insert(tk.END, "------------------------\n")
            self.chat_log.insert(tk.END, guidance + "\n\n")
            self.update_status("Answer found in local Knowledge Base.")
            self.chat_log.see(tk.END)
            return

        try:
            if mentions_vlan and not is_change:
                # Info query: list VLANs by querying device with auto-correct
                output, used_cmd = self._execute_vlan_show_with_autocorrect(manufacturer)
                self.update_status(f"Ran: {used_cmd}")
                vlans = self._parse_vlans(manufacturer, output)
                summary = self._summarize_vlans(vlans)
                self.chat_log.insert(tk.END, f"Device: {summary}\n")
                if vlans:
                    for v in vlans:
                        name = v.get('name') or '-'
                        self.chat_log.insert(tk.END, f"- VLAN {v['id']} Name: {name}\n")
                self.chat_log.insert(tk.END, "\n")
                self.chat_log.see(tk.END)
                self.update_status("Chat: VLANs listed.")
                return
        except Exception as e:
            self.chat_log.insert(tk.END, f"System: Failed to query device: {e}\n\n")
            self.update_status("Chat query failed")
            return

        # Change request: generate commands with AI
        self.set_busy(True, "Generating commands via AI…")
        # Helper: get available commands from editor or DB fallback
        def _get_available_commands_context_for_ai():
            try:
                text = self.available_commands_text.get('1.0', tk.END).strip()
            except Exception:
                text = ''
            if text:
                return text
            # Fallback: latest cached available commands for this manufacturer
            try:
                if self.db_conn:
                    cursor = self.db_conn.cursor()
                    cursor.execute(
                        "SELECT commands_text FROM available_commands_cache WHERE manufacturer = ? ORDER BY timestamp DESC LIMIT 1",
                        (manufacturer,)
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        return row[0]
            except Exception:
                pass
            return ''

        commands = self.ai_provider.get_commands(
            text,
            manufacturer,
            self.model_entry.get(),
            self.ver_entry.get(),
            device_type=self.type_entry.get(),
            running_config=self.running_config_text.get('1.0', tk.END),
            available_commands=_get_available_commands_context_for_ai(),
            use_web_search=self.use_web_search_var.get(),
            ollama_model=self.ollama_model_combo.get(),
            gemini_model=self.get_selected_gemini_model_full(),
            prompt_style='default'  # First, try to get commands
        )
        self.set_busy(False)

        # If the first attempt returns an error, automatically switch to guidance mode and retry
        if commands and commands[0].strip().startswith("# AI Error:"):
            self.update_status("Request is complex, asking AI for guidance...")
            self.chat_log.insert(tk.END, f"AI: {commands[0]}\n")
            self.chat_log.insert(tk.END, "\nAI: The request is complex. Here is some general guidance:\n\n")

            self.set_busy(True, "Asking AI for guidance…")
            guidance_response = self.ai_provider.get_commands(
                text,
                manufacturer,
                self.model_entry.get(),
                self.ver_entry.get(),
                device_type=self.type_entry.get(),
                running_config=self.running_config_text.get('1.0', tk.END),
                available_commands=_get_available_commands_context_for_ai(),
                use_web_search=self.use_web_search_var.get(),
                ollama_model=self.ollama_model_combo.get(),
                gemini_model=self.get_selected_gemini_model_full(),
                prompt_style='guidance'  # Retry in guidance mode
            )
            self.set_busy(False)
            # Display guidance and clear the other panes as there are no commands to push
            for line in guidance_response:
                self.chat_log.insert(tk.END, line + "\n")
            self.chat_log.insert(tk.END, "\n")
            self.ai_output.delete('1.0', tk.END)
            self.last_chat_response = guidance_response # Save for KB

        else:  # Success on the first try, we have commands
            self.chat_log.insert(tk.END, "AI: Proposed commands:\n")
            for cmd in commands:
                # Insert command text
                self.chat_log.insert(tk.END, cmd + "\n")
                # Append a per-command Send-to-CLI button
                try:
                    btn = tk.Button(self.chat_log, text="Send to CLI", command=lambda c=cmd: self._send_cli_from_chat(c))
                    self.chat_log.window_create(tk.END, window=btn)
                    # Add an Edit & Save Correction button next to it
                    edit_btn = tk.Button(self.chat_log, text="Edit & Save", command=lambda c=cmd: self._edit_and_save_chat_command(c))
                    self.chat_log.window_create(tk.END, window=edit_btn)
                    self.chat_log.insert(tk.END, "\n")
                except Exception:
                    # Fallback: just note the action if button cannot be created
                    self.chat_log.insert(tk.END, "(Use Sync Commands or copy-paste to CLI)\n")
            self.chat_log.insert(tk.END, "\n")
            self.last_chat_response = commands # Save for KB
            # Save successful, non-error commands to the database
            if commands and not any("# AI Error:" in cmd for cmd in commands):
                self._save_commands_to_db(manufacturer, text, commands)

        # Do not mirror into AI push pane; keep Push-to-Device disabled for chat output
        try:
            self.ai_output.delete('1.0', tk.END)
        except Exception:
            pass
        self.chat_log.see(tk.END)
        self.update_status("Chat: commands generated")

        self.fix_command_btn.configure(state=tk.DISABLED) # Disable after use

    def _send_cli_from_chat(self, cmd):
        """Send a single command from chat output to the connected device CLI."""
        try:
            if not self.connection or not getattr(self, 'is_connected', False):
                messagebox.showerror("Error", "Not connected to any device.")
                return
            if not cmd or cmd.strip().startswith('#'):
                return
            self._send_command(cmd.strip())
        except Exception as e:
            try:
                self.log_to_terminal(f"Error sending command: {e}", "error")
            except Exception:
                pass

    def _edit_and_save_chat_command(self, incorrect_cmd):
        """Allow the user to edit an AI-proposed command and save the correction.

        - Opens a prompt to edit the command.
        - Saves manufacturer-scoped correction to local DB.
        - Offers to send the corrected command immediately.
        """
        try:
            if not incorrect_cmd:
                return
            new_cmd = simpledialog.askstring(
                "Edit Command",
                "Edit the command and click OK to save as a correction:",
                initialvalue=incorrect_cmd
            )
            if new_cmd is None:
                return
            new_cmd = new_cmd.strip()
            if not new_cmd:
                return
            # Persist correction scoped by current manufacturer/device type
            try:
                self._save_correction_to_db(incorrect_cmd, new_cmd)
                self.chat_log.insert(tk.END, f"Saved correction: '{incorrect_cmd}' -> '{new_cmd}'\n")
            except Exception:
                pass

            # Optionally send the corrected command now
            try:
                if messagebox.askyesno("Send Now?", "Send corrected command to device now?"):
                    self._send_cli_from_chat(new_cmd)
            except Exception:
                pass
        except Exception as e:
            try:
                self.log_to_terminal(f"Failed to edit/save correction: {e}", "error")
            except Exception:
                pass

    def sync_chat_commands(self):
        """Send the last chat response commands to the device in sequence."""
        if not self.connection or not getattr(self, 'is_connected', False):
            messagebox.showerror("Error", "Not connected to any device.")
            return

        commands = getattr(self, 'last_chat_response', None)
        if not commands:
            messagebox.showinfo("Info", "No chat commands available to sync.")
            return

        # Filter out comments and blanks
        to_send = [c.strip() for c in commands if c and not c.strip().startswith('#')]
        if not to_send:
            messagebox.showinfo("Info", "No valid commands to sync.")
            return

        self.log_to_terminal(f"\n>>> Syncing {len(to_send)} commands from Chat...", "info")
        self.set_busy(True, f"Syncing {len(to_send)} commands…")
        self._pause_serial_reader()
        try:
            for cmd in to_send:
                # Reuse existing single-command send behavior
                self._send_command(cmd)
                # Brief pause to avoid overruns
                time.sleep(0.2)
        except Exception as e:
            self.log_to_terminal(f"Sync error: {e}", "error")
        finally:
            try:
                self._resume_serial_reader()
            except Exception:
                pass
            self.set_busy(False)
            self.update_status("Sync complete")

    def enter_privileged_or_config_mode(self):
        """Enter vendor-appropriate privileged/config mode.

        - Cisco: send 'enable' and then the Enable password.
        - H3C/Huawei: send 'system-view'.
        - Juniper: send 'configure'.
        """
        if not self.connection or not getattr(self, "is_connected", False):
            messagebox.showerror("Error", "Not connected to any device.")
            return

        manufacturer = (self.man_entry.get() or '').strip().lower()
        enable_password = (self.enable_pass_entry.get() or '')

        try:
            if 'cisco' in manufacturer:
                # Cisco enable
                self.log_to_terminal("\n> enable\r\n", "command")
                self.connection.write(b"enable\r\n")
                time.sleep(0.4)
                if enable_password:
                    self.log_to_terminal("> ********\r\n", "command")
                    self.connection.write((enable_password + "\r\n").encode())
            elif 'h3c' in manufacturer or 'huawei' in manufacturer:
                # H3C/Huawei VRP
                self.log_to_terminal("\n> system-view\r\n", "command")
                self.connection.write(b"system-view\r\n")
            elif 'juniper' in manufacturer:
                # Junos
                self.log_to_terminal("\n> configure\r\n", "command")
                self.connection.write(b"configure\r\n")
            else:
                # Fallback: try 'enable' then 'system-view'
                self.log_to_terminal("\n> enable (fallback)\r\n", "command")
                self.connection.write(b"enable\r\n")
                time.sleep(0.3)
                if enable_password:
                    self.connection.write((enable_password + "\r\n").encode())
                time.sleep(0.3)
                self.log_to_terminal("> system-view (fallback)\r\n", "command")
                self.connection.write(b"system-view\r\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to enter mode: {e}")

    def save_device_config(self):
        """Run the vendor-appropriate save/commit command to persist configuration."""
        if not self.connection or not getattr(self, "is_connected", False):
            messagebox.showerror("Error", "Not connected to any device.")
            return

        manufacturer = (self.man_entry.get() or '').strip().lower()
        self.set_busy(True, "Saving device configuration…")
        try:
            if 'cisco' in manufacturer:
                # Use 'write memory' for IOS
                self.log_to_terminal("\n> write memory\r\n", "command")
                self.connection.write(b"write memory\r\n")
            elif 'h3c' in manufacturer or 'huawei' in manufacturer:
                # H3C/Huawei typically 'save' with confirmation
                self.log_to_terminal("\n> save\r\n", "command")
                self.connection.write(b"save\r\n")
                time.sleep(0.4)
                # Attempt to confirm automatically
                try:
                    self.connection.write(b"y\r\n")
                except Exception:
                    pass
            elif 'juniper' in manufacturer:
                # Junos commit
                self.log_to_terminal("\n> commit\r\n", "command")
                self.connection.write(b"commit\r\n")
            else:
                # Generic attempt
                self.log_to_terminal("\n> save (generic)\r\n", "command")
                self.connection.write(b"save\r\n")
                time.sleep(0.4)
                try:
                    self.connection.write(b"y\r\n")
                except Exception:
                    pass
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
        finally:
            self.set_busy(False)

    def fetch_running_config(self):
        if not self.connection or not hasattr(self, 'is_connected') or not self.is_connected:
            messagebox.showerror("Error", "Not connected to any device.")
            return

        manufacturer = self.man_entry.get().lower().strip()
        if not manufacturer:
            messagebox.showerror("Input Error", "Manufacturer is required to fetch running config.")
            return

        # Determine the correct command
        if manufacturer == 'h3c':
            cmd = 'display current-configuration'
        else:
            # A common default for many vendors like Cisco, Arista
            cmd = 'show running-config'

        self.update_status(f"Fetching running-config with '{cmd}'...")
        self.log_to_terminal(f"\n>>> Fetching running-config with '{cmd}'... (this may take a moment)", "info")
        
        self.set_busy(True, "Fetching running config…")
        try:
            # Use a long timeout to capture the entire configuration
            config = self.run_device_command(cmd, timeout=30)
            self.running_config_text.delete('1.0', tk.END)
            self.running_config_text.insert(tk.END, config)
            self.update_status("Running config fetched successfully.")
            self.log_to_terminal("Running config fetched and added to AI context.", "info")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch running-config: {e}")
            self.update_status("Failed to fetch running-config.")
        finally:
            self.set_busy(False)

    def fetch_available_commands(self):
        if not self.connection or not hasattr(self, 'is_connected') or not self.is_connected:
            messagebox.showerror("Error", "Not connected to any device.")
            return

        self.update_status("Fetching available commands with '?'...")
        self.log_to_terminal("\n>>> Fetching available commands with '?'...", "info")
        
        self.set_busy(True, "Fetching available commands…")
        try:
            # Use a long timeout to capture potentially paginated output
            commands_output = self.run_device_command('?', timeout=20)
            self.available_commands_text.delete('1.0', tk.END)
            self.available_commands_text.insert(tk.END, commands_output)
            self.update_status("Available commands fetched.")
            self.log_to_terminal("Available commands fetched and added to AI context.", "info")

            # Try to determine context from the prompt to save to KB
            last_line = self.terminal.get("end-2l", "end-1l").strip()
            category = "Available Commands"
            if last_line.startswith('[') and last_line.endswith(']'):
                category += " - System View"
            elif last_line.startswith('<') and last_line.endswith('>'):
                category += " - User View"
            
            # Append to CLI DB if opted-in
            try:
                if getattr(self, 'append_available_to_db_var', None) and self.append_available_to_db_var.get():
                    self._save_available_commands_to_db(self.man_entry.get(), category, commands_output)
                    self.update_status("Appended '?' output to CLI DB")
            except Exception as e:
                self.log_to_terminal(f"Append to DB failed: {e}", "error")

            # Also save into Knowledge Base for quick lookup
            self._save_knowledge_to_db(self.man_entry.get(), category, commands_output.split('\n'))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch available commands: {e}")
            self.update_status("Failed to fetch available commands.")
        finally:
            self.set_busy(False)

    def export_running_config_to_txt(self):
        """Export the current running config text area to a .txt file."""
        try:
            text = self.running_config_text.get('1.0', tk.END)
            if not text.strip():
                try:
                    messagebox.showinfo("Export", "Running config is empty. Fetch it first.")
                except Exception:
                    pass
                return

            # Determine suggested filename from device hostname/sysname
            def _extract_hostname_from_config(cfg_text):
                try:
                    for line in cfg_text.splitlines()[:200]:
                        m = re.search(r"^\s*(hostname|sysname)\s+([A-Za-z0-9._-]+)", line, re.IGNORECASE)
                        if m:
                            return m.group(2).strip()
                    m = re.search(r"host-?name\s+([A-Za-z0-9._-]+)", cfg_text, re.IGNORECASE)
                    if m:
                        return m.group(1).strip()
                except Exception:
                    pass
                return ""

            def _extract_hostname_from_prompt():
                try:
                    last_line = self.terminal.get("end-2l", "end-1l").strip()
                    if last_line.startswith('[') and last_line.endswith(']'):
                        return last_line.strip('[]').strip()
                    if last_line.startswith('<') and last_line.endswith('>'):
                        return last_line.strip('<>').strip()
                    m = re.search(r"([A-Za-z0-9._-]+)\s*[#>]\s*$", last_line)
                    if m:
                        return m.group(1)
                except Exception:
                    pass
                return ""

            def _sanitize_filename(name):
                try:
                    name = (name or "").strip()
                    if not name:
                        return ""
                    name = re.sub(r"[<>:\"/\\|?*]", "", name)
                    name = name.strip('. ')
                    return name
                except Exception:
                    return ""

            hostname = _extract_hostname_from_config(text) or _extract_hostname_from_prompt()
            hostname = _sanitize_filename(hostname)
            suggested_filename = f"{hostname}.txt" if hostname else "running-config.txt"

            path = filedialog.asksaveasfilename(
                title="Save Running Config",
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                initialfile=suggested_filename
            )
            if not path:
                return

            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            self.update_status(f"Running config exported to {os.path.basename(path)}")
            self.log_to_terminal(f"Running config exported to: {path}", "info")
            try:
                messagebox.showinfo("Export Complete", f"Saved to: {path}")
            except Exception:
                pass
        except Exception as e:
            try:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
            except Exception:
                pass
            self.log_to_terminal(f"Export failed: {e}", "error")

    def fetch_device_info(self):
        if not self.connection or not hasattr(self, 'is_connected') or not self.is_connected:
            messagebox.showerror("Error", "Not connected to any device.")
            return

        manufacturer = self.man_entry.get().lower().strip()
        if not manufacturer:
            messagebox.showerror("Input Error", "Manufacturer is required to fetch device info.")
            return

        # Clear old info
        self.model_entry.delete(0, tk.END)
        self.ver_entry.delete(0, tk.END)

        # H3C-specific flow: send RETURN first, detect mode, then run manuinfo
        if manufacturer == 'h3c':
            try:
                self.log_to_terminal("\n>>> H3C detected: sending ENTER to refresh prompt...", "info")
                self.send_enter_key()
                # Small delay to allow prompt to render, then detect mode
                def detect_and_run():
                    try:
                        last_line = self.terminal.get("end-2l", "end-1l").strip()
                        mode = "Unknown"
                        if last_line.startswith('[') and last_line.endswith(']'):
                            mode = "System View"
                        elif last_line.startswith('<') and last_line.endswith('>'):
                            mode = "User View"
                        self.log_to_terminal(f"Detected prompt mode: {mode}", "info")
                    except Exception:
                        pass

                    cmd = 'display device manuinfo'
                    self.log_to_terminal(f"\n>>> Fetching device info with '{cmd}'...", "info")
                    self.device_info_attempt = 1
                    # Send and then parse after a short delay
                    try:
                        self.connection.write((cmd + '\r\n').encode())
                        self.update_status("Fetching device info... please wait 3 seconds.")
                        self.after(3000, lambda: self.parse_device_info(cmd))
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to send command: {e}")
                        self.log_to_terminal(f"\n[Send error] {e}\n", "error")
                        self.update_status("Error fetching device info.")

                self.after(400, detect_and_run)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to prepare H3C prompt: {e}")
                self.update_status("Error preparing H3C prompt.")
            return

        # Non-H3C vendors: use typical version commands
        cmd = 'show version'
        self.log_to_terminal(f"\n>>> Fetching device info with '{cmd}'...", "info")
        self.device_info_attempt = 1
        try:
            self.connection.write((cmd + '\r\n').encode())
            self.update_status("Fetching device info... please wait 3 seconds.")
            self.after(3000, lambda: self.parse_device_info(cmd))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send command: {e}")
            self.log_to_terminal(f"\n[Send error] {e}\n", "error")
            self.update_status("Error fetching device info.")

    def parse_device_info(self, command_sent):
        """Parses the output from the terminal to find model and version."""
        self.log_to_terminal("\n>>> Parsing device info from terminal output...", "info")

        full_text = self.terminal.get("1.0", tk.END)

        # Heuristic: find the text after the last instance of the command we sent
        try:
            last_cmd_index = full_text.rindex(command_sent)
            # Look at a reasonable chunk of text after the command
            relevant_output = full_text[last_cmd_index:].splitlines()
            relevant_output = "\n".join(relevant_output[:60]) # Limit to 60 lines to capture multi-line banners
        except ValueError:
            self.log_to_terminal("Could not find command output in terminal. Is the device responsive?", "error")
            self.update_status("Failed to parse device info.")
            return

        manufacturer = self.man_entry.get().lower().strip()
        model, version = "", ""

        # Special handling for H3C 'display device manuinfo' which reports device name
        if manufacturer == 'h3c' and 'display device manuinfo' in command_sent.lower():
            name_match = re.search(r"DEVICE_NAME\s*[:=]\s*([^\n]+)", relevant_output, re.IGNORECASE)
            if name_match:
                model = name_match.group(1).strip()
            vendor_match = re.search(r"VENDOR_NAME\s*[:=]\s*([^\n]+)", relevant_output, re.IGNORECASE)
            if vendor_match:
                try:
                    self.man_entry.set(vendor_match.group(1).strip())
                except Exception:
                    pass
            # manuinfo doesn't include software version; follow up with 'display version'
            if not model or not version:
                if getattr(self, 'device_info_attempt', 1) < 3:
                    try:
                        self.connection.write(('display version\r\n').encode())
                        self.device_info_attempt = 3
                        # If we already parsed model, reflect it immediately
                        if model:
                            self.model_entry.delete(0, tk.END)
                            self.model_entry.insert(0, model)
                            self.log_to_terminal(f"Found Model: {model}", "info")
                        self.update_status("Fetching version... please wait 3 seconds.")
                        self.after(3000, lambda: self.parse_device_info('display version'))
                        return
                    except Exception as e:
                        self.log_to_terminal(f"Follow-up version fetch failed: {e}", "error")

        # --- Regex patterns for parsing ---
        patterns = {
            'h3c': {
                # Multiple patterns to capture H3C model
                'model_patterns': [
                    re.compile(r"H3C\s+([A-Z0-9-]+)\s+uptime", re.IGNORECASE),
                    re.compile(r"\[Subslot\s*\d+\]\s*([A-Z0-9-]+)\s+Hardware", re.IGNORECASE),
                    re.compile(r"H3C\s+([A-Z0-9-]+)\s", re.IGNORECASE),
                ],
                'version_patterns': [
                    re.compile(r"Comware\s+Software,\s+Version\s+([^\n]+)", re.IGNORECASE),
                    re.compile(r"System\s+image\s+version:\s*([^\n]+)", re.IGNORECASE),
                    re.compile(r"Boot\s+image\s+version:\s*([^\n]+)", re.IGNORECASE),
                ],
            },
            'cisco': {
                'model': re.compile(r"cisco\s+([\w\S-]+)"),
                'version': re.compile(r"Cisco IOS Software.*, Version\s+([\w\d.()SE]+)")
            },
            'juniper': {
                'model': re.compile(r"Model:\s+([\w-]+)"),
                'version': re.compile(r"Junos:\s+([\d\.\w-]+)")
            }
        }

        # Use specific patterns if available
        if manufacturer in patterns:
            if manufacturer == 'h3c':
                for rx in patterns['h3c']['model_patterns']:
                    mm = rx.search(relevant_output)
                    if mm:
                        model = mm.group(1).strip()
                        break
                for rxv in patterns['h3c']['version_patterns']:
                    vm = rxv.search(relevant_output)
                    if vm:
                        version = vm.group(1).strip()
                        break
            else:
                model_match = patterns[manufacturer]['model'].search(relevant_output)
                if model_match: model = model_match.group(1).strip()
                version_match = patterns[manufacturer]['version'].search(relevant_output)
                if version_match: version = version_match.group(1).strip()

        # Generic fallback patterns if specific ones fail
        if not model:
            # Look for lines starting with "Model", "Hardware", etc.
            generic_model_match = re.search(r"^(?:Model|Hardware|Chassis|PID)\s*[:=]\s*([\w\S-]+)", relevant_output, re.MULTILINE | re.IGNORECASE)
            if generic_model_match: model = generic_model_match.group(1).strip()

        if not version:
            generic_version_match = re.search(r"(?:Version|release|ROM|SW Version)\s+([\d\w.()-]+)", relevant_output, re.IGNORECASE)
            if generic_version_match: version = generic_version_match.group(1).strip()

        # If H3C and still no model, try an alternate command for manufacturer info
        if manufacturer == 'h3c' and not model:
            if getattr(self, 'device_info_attempt', 1) < 2:
                try:
                    alt_cmd = 'display device manuinfo'
                    self.log_to_terminal(f"\nModel not found; trying alternate H3C command '{alt_cmd}'...", "info")
                    self.connection.write((alt_cmd + '\r\n').encode())
                    self.device_info_attempt = 2
                    self.after(3000, lambda: self.parse_device_info(alt_cmd))
                    return
                except Exception as e:
                    self.log_to_terminal(f"Alternate command failed: {e}", "error")

        # Update UI
        if model:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, model)
            self.log_to_terminal(f"Found Model: {model}", "info")
        else:
            self.log_to_terminal("Could not parse model from output.", "error")

        if version:
            self.ver_entry.delete(0, tk.END)
            self.ver_entry.insert(0, version)
            self.log_to_terminal(f"Found Version: {version}", "info")
        else:
            self.log_to_terminal("Could not parse version from output.", "error")

        if model and version:
            self.update_status("Device info fetched successfully.")
        else:
            self.update_status("Could not fetch all device info.")

        # Persist parsed device info to the currently selected profile for continuous configuration
        try:
            self._persist_device_info_to_profile()
        except Exception as e:
            # Non-fatal; log and continue
            self.log_to_terminal(f"Profile persistence skipped: {e}", "error")

    def save_chat_to_knowledge(self):
        """Saves the last AI chat response to the knowledge base after asking for a category."""
        if not self.last_chat_response:
            messagebox.showinfo("Save to KB", "There is no AI response to save.")
            return
        manufacturer = self.man_entry.get().strip()
        if not manufacturer:
            messagebox.showerror("Input Error", "Manufacturer must be set to save to the knowledge base.")
            return

        category = simpledialog.askstring("Save to Knowledge Base", "Enter a category for this information:")
        if not category or not category.strip():
            messagebox.showinfo("Save to KB", "Save cancelled because no category was provided.")
            return
        
        try:
            self._save_knowledge_to_db(manufacturer, category.strip(), self.last_chat_response)
            messagebox.showinfo("Success", f"Saved to knowledge base under category: '{category.strip()}'")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save to knowledge base: {e}")

    def set_ai_provider(self):
        provider = self.ai_provider_combo.get()
        api_key = self.api_key_entry.get()
        if provider in ["Gemini", "OpenAI"] and not api_key:
            messagebox.showerror("Error", "API Key is required for this provider.")
            return
        self.ai_provider.set_provider(provider, api_key)
        messagebox.showinfo("Success", f"AI Provider set to {provider}.")
        self.update_status(f"AI Provider set to {provider}")

    def push_ai_commands(self):
        if not self.connection or not self.is_connected:
            messagebox.showerror("Error", "Not connected to any device.")
            return
        
        commands_text = self.ai_output.get("1.0", tk.END)
        lines = commands_text.split("------------------------")
        if len(lines) < 2:
            messagebox.showinfo("Info", "No commands to push.")
            return
        
        original_commands = [cmd.strip() for cmd in lines[1].strip().split("\n") if cmd.strip() and not cmd.startswith("#")]
        
        if not original_commands:
            messagebox.showinfo("Info", "No valid commands found to push.")
            return

        self.log_to_terminal(f"\n>>> Pushing {len(original_commands)} commands from AI Assistant...", "info")
        self.set_busy(True, f"Sending {len(original_commands)} commands…")

        self._pause_serial_reader()
        try:
            correction_failed_permanently = False
            for cmd in original_commands:
                self.log_to_terminal(f"\n> {cmd}", "command")
                self.connection.write((cmd + '\r\n').encode())
                
                response = ""
                start_time = time.time()
                pager_re = re.compile(r"\s*-{2,}\s*More\s*-{2,}\s*|--More--|----\s*More\s*----", re.IGNORECASE)
                while time.time() - start_time < 3:
                    if self.connection.in_waiting > 0:
                        chunk = self.connection.read(self.connection.in_waiting).decode('utf-8', errors='ignore')
                        response += chunk
                        if pager_re.search(response):
                            # advance pager
                            self.connection.write(b' ')
                            response = pager_re.sub('', response)
                            start_time = time.time()
                    time.sleep(0.1)
                
                if response:
                    self.log_to_terminal(response, "output")

                if self._is_cli_error(response):
                    self.log_to_terminal("--- ERROR DETECTED! Attempting automatic correction... ---", "error")
                    
                    correction_list = self._get_ai_correction(cmd, response)

                    is_fix = not (correction_list and correction_list[0].strip().startswith("#"))

                    if not is_fix:
                        self.log_to_terminal("--- AI could not find a correction. Halting command push. ---", "error")
                        self.ai_output.delete('1.0', tk.END)
                        self.ai_output.insert(tk.END, "AI could not determine a fix. Last response:\n" + "\n".join(correction_list))
                        break

                    self.log_to_terminal(f"--- AI suggests correction: {' '.join(correction_list)} ---", "info")
                    
                    correction_succeeded = True
                    for fix_cmd in correction_list:
                        self.log_to_terminal(f"\n> {fix_cmd} (auto-correction)", "command")
                        self.connection.write((fix_cmd + '\r\n').encode())
                        fix_response = ""
                        fix_start_time = time.time()
                        pager_re2 = re.compile(r"\s*-{2,}\s*More\s*-{2,}\s*|--More--|----\s*More\s*----", re.IGNORECASE)
                        while time.time() - fix_start_time < 3:
                            if self.connection.in_waiting > 0:
                                chunk2 = self.connection.read(self.connection.in_waiting).decode('utf-8', errors='ignore')
                                fix_response += chunk2
                                if pager_re2.search(fix_response):
                                    self.connection.write(b' ')
                                    fix_response = pager_re2.sub('', fix_response)
                                    fix_start_time = time.time()
                            time.sleep(0.1)
                        
                        if fix_response:
                            self.log_to_terminal(fix_response, "output")

                        if self._is_cli_error(fix_response):
                            self.log_to_terminal("--- AI CORRECTION FAILED! Halting command push. ---", "error")
                            messagebox.showerror("Correction Failed", "The AI-suggested correction also failed. Halting all operations.")
                            correction_succeeded = False
                            correction_failed_permanently = True
                            break
                    
                    if correction_failed_permanently:
                        break

                    if correction_succeeded:
                        self.log_to_terminal("--- Correction successful. Continuing with next command. ---", "info")
                        self._save_correction_to_db(cmd, '\n'.join(correction_list))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send commands: {str(e)}")
            self.log_to_terminal(f"Error sending commands: {str(e)}", "error")
        finally:
            self._resume_serial_reader()
            self.set_busy(False)
            
        self.update_status("Command push finished or was halted by an error.")

    def log_to_terminal(self, message, tag=None):
        self.terminal.tag_config("info", foreground="cyan")
        self.terminal.tag_config("error", foreground="red")
        self.terminal.tag_config("command", foreground="yellow")
        self.terminal.tag_config("prompt", foreground="lime green")

        # Ensure device/log output appears ABOVE the current prompt/input line
        insert_index = tk.END
        try:
            if hasattr(self.terminal, 'mark_set'):
                # Insert before prompt mark if it exists
                try:
                    insert_index = self.terminal.index('PROMPT')
                except Exception:
                    insert_index = tk.END
        except Exception:
            insert_index = tk.END
        self.terminal.insert(insert_index, message + "\n", tag)
        self.terminal.see(tk.END)
        self.update_idletasks()

    def _persist_device_info_to_profile(self):
        """Persist manufacturer, model, and version to the active profile for memory."""
        profile_name = self.profile_combo.get()
        if not profile_name:
            return
        # Gather current fields
        manufacturer = self.man_entry.get()
        model = self.model_entry.get()
        version = self.ver_entry.get()
        if not (manufacturer or model or version):
            return
        # Update profile in memory
        profile = self.profiles.get(profile_name, {})
        profile.update({
            'com_port': self.com_port_combo.get() or profile.get('com_port', ''),
            'username': self.user_entry.get() or profile.get('username', ''),
            'password': self.pass_entry.get() or profile.get('password', ''),
            'manufacturer': manufacturer,
            'model': model,
            'version': version,
        })
        self.profiles[profile_name] = profile
        # Write to disk
        try:
            with open(self.profiles_file, 'w') as f:
                json.dump(self.profiles, f, indent=4)
            self.log_to_terminal(f"Profile '{profile_name}' updated with device info.", "info")
        except Exception as e:
            self.log_to_terminal(f"Failed to update profile file: {e}", "error")

    def load_profiles(self):
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, 'r') as f:
                    self.profiles = json.load(f)
            except json.JSONDecodeError:
                backup_file = self.profiles_file + '.bak'
                try:
                    os.rename(self.profiles_file, backup_file)
                    messagebox.showwarning(
                        "Profile Load Error",
                        f"The profile file '{self.profiles_file}' was corrupted.\n\n"
                        f"It has been renamed to '{backup_file}'.\n"
                        "The application will start with fresh profiles."
                    )
                except Exception as e:
                    messagebox.showerror(
                        "Profile Load Error",
                        f"The profile file '{self.profiles_file}' is corrupted and a backup could not be created: {e}"
                    )
                self.profiles = {}
        self.update_profile_list()

    def update_profile_list(self):
        profile_names = list(self.profiles.keys())
        try:
            self.profile_combo.configure(values=profile_names)
        except Exception:
            pass
        if profile_names:
            self.profile_combo.set(profile_names[0])
            self.load_selected_profile()

    def load_selected_profile(self, event=None):
        profile_name = self.profile_combo.get()
        if profile_name in self.profiles:
            profile = self.profiles[profile_name]
            self.com_port_combo.set(profile.get('com_port', ''))
            self.user_entry.delete(0, tk.END); self.user_entry.insert(0, profile.get('username', ''))
            self.pass_entry.delete(0, tk.END); self.pass_entry.insert(0, profile.get('password', ''))
            self.enable_pass_entry.delete(0, tk.END); self.enable_pass_entry.insert(0, profile.get('enable_password', ''))
            try:
                self.man_entry.set(profile.get('manufacturer', ''))
            except Exception:
                pass
            # New: connection type, host, and port
            try:
                self.conn_type_var.set(profile.get('conn_type', self.conn_type_var.get() or 'Serial'))
            except Exception:
                pass
            try:
                self.host_entry.delete(0, tk.END); self.host_entry.insert(0, profile.get('host', ''))
            except Exception:
                pass
            try:
                self.port_entry.delete(0, tk.END); self.port_entry.insert(0, str(profile.get('port', '')))
            except Exception:
                pass
            self.type_entry.delete(0, tk.END); self.type_entry.insert(0, profile.get('device_type', ''))
            self.model_entry.delete(0, tk.END); self.model_entry.insert(0, profile.get('model', ''))
            self.ver_entry.delete(0, tk.END); self.ver_entry.insert(0, profile.get('version', ''))
            try:
                self.running_config_text.delete('1.0', tk.END)
                self.running_config_text.insert(tk.END, profile.get('running_config', ''))
            except Exception:
                pass
            try:
                self.available_commands_text.delete('1.0', tk.END)
                self.available_commands_text.insert(tk.END, profile.get('available_commands', ''))
            except Exception:
                pass
            self.update_status(f"Loaded profile: {profile_name}")

    def save_profile(self):
        profile_name = simpledialog.askstring("Save Profile", "Enter a name for this profile:")
        if profile_name:
            self.profiles[profile_name] = {
                'com_port': self.com_port_combo.get(),
                'username': self.user_entry.get(),
                'password': self.pass_entry.get(),
                'enable_password': self.enable_pass_entry.get(),
                'manufacturer': self.man_entry.get(),
                'device_type': self.type_entry.get(),
                'model': self.model_entry.get(),
                'version': self.ver_entry.get(),
                'running_config': self.running_config_text.get('1.0', tk.END),
                'available_commands': self.available_commands_text.get('1.0', tk.END),
                # New network fields
                'conn_type': self.conn_type_var.get(),
                'host': self.host_entry.get(),
                'port': int(self.port_entry.get() or '0') if (self.port_entry.get() or '').strip().isdigit() else (self.port_entry.get() or '')
            }
            with open(self.profiles_file, 'w') as f:
                json.dump(self.profiles, f, indent=4)
            self.update_profile_list()
            self.profile_combo.set(profile_name)
            self.update_status(f"Saved profile: {profile_name}")

    def update_profile(self):
        profile_name = self.profile_combo.get()
        if not profile_name:
            messagebox.showerror("Error", "No profile selected to update.")
            return

        if messagebox.askyesno("Confirm Update", f"Are you sure you want to overwrite the profile '{profile_name}'?"):
            self.profiles[profile_name] = {
                'com_port': self.com_port_combo.get(),
                'username': self.user_entry.get(),
                'password': self.pass_entry.get(),
                'enable_password': self.enable_pass_entry.get(),
                'manufacturer': self.man_entry.get(),
                'device_type': self.type_entry.get(),
                'model': self.model_entry.get(),
                'version': self.ver_entry.get(),
                'running_config': self.running_config_text.get('1.0', tk.END),
                'available_commands': self.available_commands_text.get('1.0', tk.END),
                'conn_type': self.conn_type_var.get(),
                'host': self.host_entry.get(),
                'port': int(self.port_entry.get() or '0') if (self.port_entry.get() or '').strip().isdigit() else (self.port_entry.get() or '')
            }
            with open(self.profiles_file, 'w') as f:
                json.dump(self.profiles, f, indent=4)
            self.update_status(f"Updated profile: {profile_name}")
            messagebox.showinfo("Success", f"Profile '{profile_name}' has been updated.")

    def _search_knowledge_base(self, query):
        """Searches the local KB for an answer before asking the AI."""
        if not self.db_conn:
            return None
        manufacturer = self.man_entry.get().strip()
        if not manufacturer:
            return None
        
        try:
            cursor = self.db_conn.cursor()
            # Simple search: look for keywords in the category
            # A more advanced search could use full-text search or keyword tokenization
            search_term = f"%{query}%"
            cursor.execute(
                "SELECT guidance_text, category FROM command_knowledge WHERE manufacturer = ? AND category LIKE ? ORDER BY length(category) ASC LIMIT 1",
                (manufacturer, search_term)
            )
            result = cursor.fetchone()
            if result:
                return result # Returns (guidance_text, category)
        except Exception as e:
            self.log_to_terminal(f"Knowledge base search failed: {e}", "error")
        
        return None

    def delete_profile(self):
        profile_name = self.profile_combo.get()
        if not profile_name: return
        if messagebox.askyesno("Confirm", f"Are you sure you want to delete the profile '{profile_name}'?"):
            if profile_name in self.profiles:
                del self.profiles[profile_name]
                with open(self.profiles_file, 'w') as f: json.dump(self.profiles, f, indent=4)
                self.com_port_combo.set('')
                for entry in [self.user_entry, self.pass_entry, self.man_entry, self.model_entry, self.ver_entry, self.host_entry, self.port_entry]:
                    entry.delete(0, tk.END)
                try:
                    self.conn_type_var.set('Serial')
                except Exception:
                    pass
                self.update_profile_list()
                self.update_status(f"Deleted profile: {profile_name}")

    def enter_enable_mode(self):
        if not self.connection or not getattr(self, "is_connected", False):
            messagebox.showerror("Error", "Not connected to any device.")
            return
        
        enable_password = self.enable_pass_entry.get()
        if not enable_password:
            messagebox.showinfo("Info", "No Enable Password has been set.")
            return

        try:
            # Send 'enable' command
            self.log_to_terminal("\n> enable\r\n", "command")
            self.connection.write(b"enable\r\n")
            # Wait a moment for the password prompt
            time.sleep(0.5)
            # Send the enable password
            self.log_to_terminal("> ********\r\n", "command") # Echo masked password
            self.connection.write((enable_password + "\r\n").encode())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send enable commands: {e}")
        # Enable mode entered; KB viewer is available via the "View KB" button.

    def show_knowledge_base_window(self):
        kb_window = tk.Toplevel(self)
        kb_window.title("Knowledge Base Browser")
        kb_window.geometry("800x600")

        top_frame = tk.Frame(kb_window)
        top_frame.pack(fill='x', padx=10, pady=5)
        tk.Button(top_frame, text="Refresh", command=lambda: self._populate_kb_viewer(tree)).pack(side=tk.LEFT)
        tk.Button(top_frame, text="Import from JSON...", command=lambda: self.import_knowledge_from_json(tree)).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Delete Selected", command=lambda: self._delete_kb_entry(tree, kb_text)).pack(side=tk.LEFT)

        pane = tk.PanedWindow(kb_window, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tree_frame = tk.Frame(pane)
        tree = ttk.Treeview(tree_frame, columns=("ID", "Manufacturer", "Category", "Timestamp"), show='headings')
        tree.heading("ID", text="ID")
        tree.heading("Manufacturer", text="Manufacturer")
        tree.heading("Category", text="Category")
        tree.heading("Timestamp", text="Timestamp")
        tree.column("ID", width=50, stretch=tk.NO)
        tree.column("Manufacturer", width=100)
        tree.column("Category", width=200)
        tree.column("Timestamp", width=150)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill='y')
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pane.add(tree_frame, height=250)

        text_frame = tk.Frame(pane)
        kb_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        kb_text.pack(fill=tk.BOTH, expand=True)
        pane.add(text_frame)

        tree.selection_data = {}
        tree.bind("<<TreeviewSelect>>", lambda event: self._on_kb_select(event, tree, kb_text))

        self._populate_kb_viewer(tree)

    def _populate_kb_viewer(self, tree):
        if not self.db_conn:
            messagebox.showerror("DB Error", "Database connection is not available.")
            return
        # Clear existing items
        for i in tree.get_children():
            tree.delete(i)
        tree.selection_data = {}
        
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT id, manufacturer, category, timestamp, guidance_text FROM command_knowledge ORDER BY timestamp DESC")
        for row in cursor.fetchall():
            item_id = tree.insert("", tk.END, values=row[:4])
            tree.selection_data[item_id] = {"guidance": row[4], "timestamp": row[3]}

    def _on_kb_select(self, event, tree, text_widget):
        try:
            selected_item = tree.selection()[0]
            data = tree.selection_data.get(selected_item, {})
            guidance = data.get("guidance", "")
            timestamp = data.get("timestamp", "")
            text_widget.delete('1.0', tk.END)
            if timestamp:
                text_widget.insert(tk.END, f"Timestamp: {timestamp}\n\n")
            text_widget.insert(tk.END, guidance)
        except IndexError:
            pass # Ignore empty selection

    def _delete_kb_entry(self, tree, text_widget=None):
        try:
            selected_item = tree.selection()[0]
        except IndexError:
            messagebox.showinfo("Delete", "No KB entry selected.")
            return
        values = tree.item(selected_item, 'values')
        if not values:
            messagebox.showinfo("Delete", "No KB entry selected.")
            return
        kb_id = values[0]
        if not messagebox.askyesno("Confirm", f"Delete KB entry ID {kb_id}? This cannot be undone."):
            return
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("DELETE FROM command_knowledge WHERE id = ?", (kb_id,))
            self.db_conn.commit()
            # Refresh view and clear text
            self._populate_kb_viewer(tree)
            if text_widget:
                text_widget.delete('1.0', tk.END)
            self.update_status(f"Deleted KB entry {kb_id}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete KB entry: {e}")

if __name__ == "__main__":
    app = NetApp()
    app.mainloop()
