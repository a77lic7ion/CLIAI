import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox, ttk, filedialog
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
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    MistralClient = None
    ChatMessage = None
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
}

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
            elif self.provider == "Mistral" and MistralClient:
                self.client = MistralClient(api_key=self.api_key)
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
            You are a world-class network engineering expert AI, acting as a helpful assistant. Your task is to provide clear, detailed, and helpful guidance for network configuration tasks.

            {final_context}

            **CONTEXT:** The user is working with a **{manufacturer.upper()} {device_type.upper() if device_type else ''}** device. Your advice should be tailored to this vendor and device type.

            **YOUR DIRECTIVES:**
            1.  **BE HELPFUL AND EXPLANATORY:** The user is asking a general question. Do not just provide commands. Explain the concepts, outline the steps involved, and provide examples.
            2.  **ASK FOR DETAILS (if needed):** If the user's request is missing critical information (like IP addresses), explain what information is needed and why. Provide placeholders in your examples (e.g., `<Your_IP_Address>`).
            3.  **STRUCTURE YOUR RESPONSE:** Use formatting like lists, steps, and code blocks to make your answer easy to read and follow.
            4.  **USE MARKDOWN:** You can use Markdown for formatting. Use code blocks (```) for command examples.
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
            You are a world-class network engineering expert AI. Your one and only task is to generate precise, executable CLI commands for a specific network device.

            {final_context}

            **CRITICAL CONTEXT:** The target device is a **{manufacturer.upper()} {device_type.upper() if device_type else ''}** device. All commands you generate **MUST** use the correct syntax for this specific platform.

            **DEVICE DETAILS:** {device_context}

            **YOUR DIRECTIVES:**
            1.  **PRIORITIZE THE PLATFORM:** The user-provided manufacturer and device type ({manufacturer.upper()} {device_type.upper() if device_type else ''}) are the most important pieces of information. Your output must be 100% correct for this platform.
            2.  **COMMANDS ONLY:** Your entire response must be only the CLI commands needed to achieve the user's goal.
            3.  **NO EXPLANATIONS:** Do not add any descriptive text, apologies, or introductory sentences like "Here are the commands...".
            4.  **NO MARKDOWN:** Do not use markdown code blocks (```).
            5.  **ONE COMMAND PER LINE:** Each command must be on a new line.
            6.  **HANDLE AMBIGUITY:** If the request is unclear or cannot be fulfilled, return a single line starting with '# AI Error:' followed by a brief explanation.

            **Example for H3C:**
            User Request: create vlan 100
            Correct Response:
            system-view
            vlan 100
            quit

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
                if not MistralClient: return ["# Mistral library not installed."]
                if not self.client: return ["# Mistral client not initialized. Set provider again."]
                messages = []
                if ChatMessage:
                    messages = [
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_request),
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_request},
                    ]
                response = self.client.chat(model="mistral-large-latest", messages=messages)
                content = ""
                try:
                    choice = response.choices[0]
                    msg = getattr(choice, "message", None)
                    if msg is not None and hasattr(msg, "content"):
                        cont = msg.content
                        if isinstance(cont, str):
                            content = cont
                        elif isinstance(cont, list):
                            # Newer SDK returns list of blocks
                            content = "".join(getattr(block, "text", "") for block in cont)
                        else:
                            content = str(cont)
                    else:
                        content = str(response)
                except Exception:
                    content = str(response)
                return [line for line in content.strip().split('\n')]

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


class NetApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NetIntelli X")
        self.geometry("1000x800")

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

        # --- Main Content Frame ---
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_pane = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_pane, bd=2)
        main_pane.add(left_frame, width=600)

        conn_frame = tk.LabelFrame(left_frame, text="Connection", bd=2, relief=tk.GROOVE)
        conn_frame.pack(pady=10, padx=10, fill="x")
        tk.Label(conn_frame, text="Profile:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.profile_combo = ttk.Combobox(conn_frame, state="readonly")
        self.profile_combo.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.profile_combo.bind("<<ComboboxSelected>>", self.load_selected_profile)
        tk.Label(conn_frame, text="COM Port:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.com_port_combo = ttk.Combobox(conn_frame, state="readonly")
        self.com_port_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(conn_frame, text="Refresh", command=self.refresh_com_ports).grid(row=1, column=2, padx=5, pady=5)
        tk.Label(conn_frame, text="Baud:").grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.baud_combo = ttk.Combobox(conn_frame, state="readonly", values=["9600","19200","38400","57600","115200"], width=8)
        self.baud_combo.grid(row=1, column=4, padx=5, pady=5, sticky="w")
        self.baud_combo.set("9600")
        # Populate COM ports now that the combobox exists
        self.refresh_com_ports()
        tk.Label(conn_frame, text="Username:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.user_entry = tk.Entry(conn_frame)
        self.user_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        tk.Label(conn_frame, text="Password:").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.pass_entry = tk.Entry(conn_frame, show="*")
        self.pass_entry.grid(row=2, column=3, padx=5, pady=5, sticky="ew")

        tk.Label(conn_frame, text="Enable Pass:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.enable_pass_entry = tk.Entry(conn_frame, show="*")
        self.enable_pass_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        self.connect_btn = tk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.enable_btn = tk.Button(conn_frame, text="Enter Enable Mode", command=self.enter_enable_mode)
        self.enable_btn.grid(row=4, column=2, columnspan=2, padx=10, pady=5, sticky="ew")
        conn_frame.columnconfigure(1, weight=1)

        profile_btn_frame = tk.Frame(conn_frame)
        profile_btn_frame.grid(row=0, column=2, columnspan=2, sticky='ew')
        tk.Button(profile_btn_frame, text="Save", command=self.save_profile).pack(side=tk.LEFT, fill='x', expand=True)
        tk.Button(profile_btn_frame, text="Update", command=self.update_profile).pack(side=tk.LEFT, fill='x', expand=True)
        tk.Button(profile_btn_frame, text="Delete", command=self.delete_profile).pack(side=tk.LEFT, fill='x', expand=True)

        terminal_frame = tk.LabelFrame(left_frame, text="Device Terminal", bd=2, relief=tk.GROOVE)
        terminal_frame.pack(pady=10, padx=10, expand=True, fill="both")
        self.terminal = scrolledtext.ScrolledText(terminal_frame, wrap=tk.WORD, bg="black", fg="white", font=("Consolas", 10))
        self.terminal.pack(expand=True, fill="both")
        # Direct input: type into terminal window and press Enter to send
        self._setup_direct_terminal_input()
        self._show_prompt()

        # Inline terminal input controls
        term_input_frame = tk.Frame(terminal_frame)
        term_input_frame.pack(fill="x", padx=10, pady=6)
        try:
            self.term_input = tk.Entry(term_input_frame, font=("Consolas", 10))
            self.term_input.pack(side=tk.LEFT, fill="x", expand=True)
            tk.Button(term_input_frame, text="Send", command=self.send_terminal_input).pack(side=tk.LEFT, padx=6)
            tk.Button(term_input_frame, text="Send RETURN", command=self.send_enter_key).pack(side=tk.LEFT, padx=6)
        except Exception:
            pass

        # Make Push-to-Device accessible near the terminal as well
        term_actions = tk.Frame(terminal_frame)
        term_actions.pack(fill="x", padx=10, pady=6)
        tk.Button(term_actions, text="Push AI Commands to Device", command=self.push_ai_commands).pack(side=tk.LEFT, fill="x", expand=True)

        # --- Right-hand side layout --- 
        right_master_frame = tk.Frame(main_pane)
        main_pane.add(right_master_frame)

        # New top frame for side-by-side config sections
        top_right_frame = tk.Frame(right_master_frame)
        top_right_frame.pack(fill="x", expand=False, pady=5, padx=5)

        # AI Configuration (now on the left of the top-right frame)
        ai_config_frame = tk.LabelFrame(top_right_frame, text="AI Configuration", relief=tk.GROOVE)
        ai_config_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))

        tk.Label(ai_config_frame, text="Provider:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ai_provider_combo = ttk.Combobox(ai_config_frame, state="readonly", values=["None", "Gemini", "OpenAI", "Mistral", "Claude", "Ollama", "Simulation"])
        self.ai_provider_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.ai_provider_combo.set("Gemini")
        self.ai_provider_combo.bind("<<ComboboxSelected>>", self.on_ai_provider_change)
        tk.Label(ai_config_frame, text="API Key:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.api_key_entry = tk.Entry(ai_config_frame, show="*")
        self.api_key_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        tk.Label(ai_config_frame, text="Ollama Model:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.ollama_model_combo = ttk.Combobox(ai_config_frame, state="readonly")
        self.ollama_model_combo.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        tk.Label(ai_config_frame, text="Gemini Model:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.gemini_model_combo = ttk.Combobox(ai_config_frame, state="readonly", values=["1.5-flash", "2.0-flash", "1.5-pro"])
        self.gemini_model_combo.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.gemini_model_combo.set("2.0-flash")
        # Buttons: Set provider and Check API Key
        self.set_ai_btn = tk.Button(ai_config_frame, text="Set AI Provider", command=self.set_ai_provider)
        self.set_ai_btn.grid(row=4, column=0, padx=5, pady=5, sticky="ew")
        self.check_api_btn = tk.Button(ai_config_frame, text="Check API Key", command=self.check_api_key)
        self.check_api_btn.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        ai_config_frame.columnconfigure(1, weight=1)

        # Terminal Options (moved from left frame to top-right)
        terminal_opts_frame = tk.LabelFrame(top_right_frame, text="Terminal Options", bd=2, relief=tk.GROOVE)
        terminal_opts_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0))

        tk.Label(terminal_opts_frame, text="Wrap Mode:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.term_wrap_combo = ttk.Combobox(terminal_opts_frame, state="readonly", values=["Wrap (word)", "No wrap"], width=12)
        self.term_wrap_combo.set("Wrap (word)")
        self.term_wrap_combo.bind("<<ComboboxSelected>>", self.on_term_wrap_change)
        self.term_wrap_combo.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        font_frame = tk.Frame(terminal_opts_frame)
        font_frame.grid(row=0, column=2, sticky="w", padx=5, pady=2)
        tk.Button(font_frame, text="Font +", command=self.increase_terminal_font).pack(side=tk.LEFT)
        tk.Button(font_frame, text="Font -", command=self.decrease_terminal_font).pack(side=tk.LEFT)
        self.auto_pager_var = tk.BooleanVar(value=True)
        tk.Checkbutton(terminal_opts_frame, text="Auto-pager", variable=self.auto_pager_var).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Button(terminal_opts_frame, text="Next Page", command=self.send_pager_next).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        tk.Button(terminal_opts_frame, text="Stop Paging", command=self.send_pager_stop).grid(row=1, column=2, sticky="w", padx=5, pady=2)
        tk.Button(terminal_opts_frame, text="Disable Paging", command=self.disable_paging).grid(row=1, column=3, sticky="w", padx=5, pady=2)
        self.fix_command_btn = tk.Button(terminal_opts_frame, text="Fix with AI", command=self.ai_fix_last_command, state=tk.DISABLED)
        self.fix_command_btn.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        tk.Button(terminal_opts_frame, text="Clear", command=self.clear_terminal).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        tk.Button(terminal_opts_frame, text="Export…", command=self.export_terminal_chat).grid(row=2, column=2, sticky="w", padx=5, pady=5)

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

        ai_assistant_frame = tk.LabelFrame(right_split, text="AI Assistant", relief=tk.GROOVE)
        right_split.add(ai_assistant_frame, minsize=300)
        context_frame = tk.Frame(ai_assistant_frame)
        context_frame.pack(pady=5, padx=10, fill='x')
        tk.Label(context_frame, text="Manufacturer:").grid(row=0, column=0, sticky="w")
        self.man_entry = tk.Entry(context_frame)
        self.man_entry.grid(row=0, column=1, sticky="ew", padx=2)
        tk.Label(context_frame, text="Device Type:").grid(row=1, column=0, sticky="w")
        self.type_entry = tk.Entry(context_frame)
        self.type_entry.grid(row=1, column=1, sticky="ew", padx=2)
        tk.Label(context_frame, text="Model:").grid(row=2, column=0, sticky="w")
        self.model_entry = tk.Entry(context_frame)
        self.model_entry.grid(row=2, column=1, sticky="ew", padx=2)
        tk.Label(context_frame, text="Version:").grid(row=3, column=0, sticky="w")
        self.ver_entry = tk.Entry(context_frame)
        self.ver_entry.grid(row=3, column=1, sticky="ew", padx=2)

        self.fetch_info_btn = tk.Button(context_frame, text="Fetch Device Info", command=self.fetch_device_info)
        self.fetch_info_btn.grid(row=4, column=0, pady=5, sticky="ew")
        self.build_db_btn = tk.Button(context_frame, text="Build Cmd DB", command=self.build_command_database)
        self.build_db_btn.grid(row=4, column=1, pady=5, sticky="ew")
        self.view_kb_btn = tk.Button(context_frame, text="View KB", command=self.show_knowledge_base_window)
        self.view_kb_btn.grid(row=4, column=2, pady=5, sticky="ew")
        self.import_json_btn = tk.Button(context_frame, text="Import CLI JSON", command=self.import_cli_json)
        self.import_json_btn.grid(row=4, column=3, pady=5, sticky="ew")

        context_frame.columnconfigure(1, weight=1)
        context_frame.columnconfigure(2, weight=1)

        ttk.Separator(ai_assistant_frame, orient='horizontal').pack(fill='x', pady=5, padx=10)

        self.fetch_config_btn = tk.Button(ai_assistant_frame, text="Fetch Running Config for AI Context", command=self.fetch_running_config)
        self.fetch_config_btn.pack(pady=5, padx=10, fill="x")
        self.running_config_text = scrolledtext.ScrolledText(ai_assistant_frame, wrap=tk.WORD, height=8, font=("Consolas", 9))
        self.running_config_text.pack(pady=5, padx=10, expand=True, fill="both")

        self.fetch_q_btn = tk.Button(ai_assistant_frame, text="Fetch '?' Commands for AI Context", command=self.fetch_available_commands)
        self.fetch_q_btn.pack(pady=5, padx=10, fill="x")
        self.available_commands_text = scrolledtext.ScrolledText(ai_assistant_frame, wrap=tk.WORD, height=6, font=("Consolas", 9))
        self.available_commands_text.pack(pady=5, padx=10, expand=True, fill="both")

        ttk.Separator(ai_assistant_frame, orient='horizontal').pack(fill='x', pady=5, padx=10)
        tk.Label(ai_assistant_frame, text="Your Request:").pack(pady=5, padx=10, anchor="w")
        self.ai_input = tk.Entry(ai_assistant_frame, font=("Arial", 10))
        self.ai_input.pack(pady=5, padx=10, fill="x")
        self.ai_input.bind("<Return>", self.query_ai)
        self.use_web_search_var = tk.BooleanVar(value=True)
        self.web_search_check = tk.Checkbutton(ai_assistant_frame, text="Use Web Search (Gemini)", variable=self.use_web_search_var)
        self.web_search_check.pack(pady=5, padx=10, anchor="w")
        tk.Button(ai_assistant_frame, text="Generate Commands", command=self.query_ai).pack(pady=5, padx=10, fill="x")
        self.ai_output = scrolledtext.ScrolledText(ai_assistant_frame, wrap=tk.WORD, height=10, font=("Consolas", 10))
        self.ai_output.pack(pady=10, padx=10, expand=True, fill="both")
        self.push_to_device_btn = tk.Button(ai_assistant_frame, text="Push to Connected Device", command=self.push_ai_commands, state=tk.DISABLED)
        self.push_to_device_btn.pack(pady=10, padx=10, fill="x")

        # Chat pane: ask questions, get backend answers, and generate commands for changes
        chat_frame = tk.LabelFrame(right_split, text="Chat", relief=tk.GROOVE)
        right_split.add(chat_frame, minsize=300)
        self.chat_log = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=12, font=("Consolas", 10))
        self.chat_log.pack(pady=6, padx=10, expand=True, fill="both")
        chat_input_frame = tk.Frame(chat_frame)
        chat_input_frame.pack(fill="x", padx=10, pady=6)
        self.chat_input = tk.Entry(chat_input_frame, font=("Arial", 10))
        self.chat_input.pack(side=tk.LEFT, fill="x", expand=True)
        self.chat_input.bind("<Return>", self.chat_ask)
        tk.Button(chat_input_frame, text="Send", command=self.chat_ask).pack(side=tk.LEFT, padx=6)
        tk.Button(chat_input_frame, text="Save to KB", command=self.save_chat_to_knowledge).pack(side=tk.LEFT, padx=6)

        # Schedule sash placement for 50/50 split
        self.after(100, lambda: right_split.sash_place(0, int(right_split.winfo_width() / 2), 0))

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
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
            self.db_conn.commit()
            self.log_to_terminal("Local command cache database initialized.", "info")
        except Exception as e:
            self.db_conn = None
            self.log_to_terminal(f"Error initializing database: {e}", "error")

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
            # Run the potentially long-running task in a separate thread
            thread = threading.Thread(target=self._build_db_worker, args=(manufacturer,), daemon=True)
            thread.start()

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

            for category, question in topics.items():
                full_request = f"For a {manufacturer} device, provide {question}"
                self.log_to_terminal(f"Querying AI for category: {category}...", "info")
                self.update_status(f"Building DB: Querying for {category}...")

                guidance = self.ai_provider.get_commands(
                    full_request,
                    manufacturer,
                    self.model_entry.get(),
                    self.ver_entry.get(),
                    device_type=self.type_entry.get(),
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
                
                time.sleep(5) # Add a small delay to avoid hitting API rate limits

            db_conn.commit()
            self.log_to_terminal("Command database build finished and saved.", "info")
            self.update_status("Command DB build finished.")
        except Exception as e:
            self.log_to_terminal(f"Database build worker failed: {e}", "error")
            self.update_status("Command DB build failed.")
        finally:
            if db_conn:
                db_conn.close()

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

    def on_ai_provider_change(self, event=None):
        provider = self.ai_provider_combo.get()
        self.api_key_entry.config(state=tk.NORMAL if provider in ["Gemini", "OpenAI", "Mistral", "Claude"] else tk.DISABLED)
        self.web_search_check.config(state=tk.NORMAL if provider == "Gemini" else tk.DISABLED)
        # Enable Gemini model selection only for Gemini provider
        try:
            self.gemini_model_combo.config(state="readonly" if provider == "Gemini" else tk.DISABLED)
        except Exception:
            pass
        
        if provider == "Ollama":
            self.ollama_model_combo.config(state="readonly")
            self.fetch_ollama_models()
        else:
            self.ollama_model_combo.config(state=tk.DISABLED)
            self.ollama_model_combo['values'] = []
            self.ollama_model_combo.set('')

    def fetch_ollama_models(self):
        self.log_to_terminal("Fetching Ollama models...", "info")
        self.update_status("Fetching Ollama models...")
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3 )
            response.raise_for_status()
            models = [m['name'] for m in response.json().get('models', [])]
            if models:
                self.ollama_model_combo['values'] = models
                self.ollama_model_combo.set(models)
                self.log_to_terminal(f"Found {len(models)} Ollama models.", "info")
                self.update_status("Ollama models loaded.")
            else:
                self.ollama_model_combo['values'] = ["No models found"]
                self.ollama_model_combo.set("No models found")
                self.update_status("Ollama: No models found.")
        except requests.exceptions.Timeout:
            self.log_to_terminal("Ollama server timed out. Is it running and responsive?", "error")
            self.update_status("Ollama server timed out.")
            self.ollama_model_combo['values'] = ["Ollama server timeout"]
            self.ollama_model_combo.set("Ollama server timeout")
        except requests.exceptions.ConnectionError:
            self.log_to_terminal("Ollama server not found at http://localhost:11434.", "error" )
            self.update_status("Ollama server not found.")
            self.ollama_model_combo['values'] = ["Ollama server not found"]
            self.ollama_model_combo.set("Ollama server not found")
        except Exception as e:
            self.log_to_terminal(f"Error fetching Ollama models: {e}", "error")
            self.update_status("Error fetching Ollama models.")
            self.ollama_model_combo['values'] = ["Error fetching models"]
            self.ollama_model_combo.set("Error fetching models")

    def refresh_com_ports(self):
        """Refresh the list of available COM ports"""
        try:
            # Guard against early calls before UI is built
            if not hasattr(self, 'com_port_combo'):
                return
            ports = serial.tools.list_ports.comports()
            available_ports = [port.device for port in ports if port.device]
            self.com_port_combo['values'] = available_ports
            if available_ports and not self.com_port_combo.get():
                self.com_port_combo.set(available_ports[0])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh COM ports: {str(e)}")

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
        if not api_key:
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
        else:
            messagebox.showinfo("API Key", "Check is currently available for Gemini only.")

    def toggle_connection(self):
        if self.connection:
            self.disconnect()
        else:
            self.connect()

    def on_term_wrap_change(self, event=None):
        mode = self.term_wrap_combo.get()
        try:
            self.terminal.config(wrap=tk.NONE if mode == "No wrap" else tk.WORD)
        except Exception:
            pass

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
            current = getattr(self, 'terminal_font_size', 10)
            new_size = min(current + 1, 48)
            self.terminal_font_size = new_size
            self.terminal.configure(font=("Consolas", new_size))
        except Exception:
            pass

    def decrease_terminal_font(self):
        # Decrease font size down to a sensible minimum
        try:
            current = getattr(self, 'terminal_font_size', 10)
            new_size = max(current - 1, 6)
            self.terminal_font_size = new_size
            self.terminal.configure(font=("Consolas", new_size))
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
        com_port = self.com_port_combo.get()
        try:
            baudrate = int(self.baud_combo.get()) if hasattr(self, 'baud_combo') and self.baud_combo.get() else 9600
        except Exception:
            baudrate = 9600
        username = self.user_entry.get()
        password = self.pass_entry.get()
        
        if not com_port:
            messagebox.showerror("Error", "Please select a COM port")
            return
        
        try:
            # Create serial connection
            self.connection = serial.Serial(
                port=com_port,
                baudrate=baudrate,
                timeout=0,          # Non-blocking reads
                write_timeout=3      # Give writes more time
            )
            # Clear buffers and start reader
            self.connection.reset_input_buffer()
            self.connection.reset_output_buffer()
            
            self.is_connected = True
            self.connect_btn.config(text="Disconnect")
            self.push_to_device_btn.config(state=tk.NORMAL)
            self.log_to_terminal(f"Connected to {com_port}")
            self.update_status(f"Connected to {com_port}")
            self.run_precheck()
            self._resume_serial_reader()
            
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect to {com_port}: {str(e)}")

    def disconnect(self):
        if self.connection:
            try:
                self._pause_serial_reader()
                self.connection.close()
                self.connection = None
                self.is_connected = False
                self.connect_btn.config(text="Connect")
                self.push_to_device_btn.config(state=tk.DISABLED)
                self.log_to_terminal("Disconnected")
                self.update_status("Disconnected")
            except Exception as e:
                messagebox.showerror("Disconnection Error", f"Error during disconnection: {str(e)}")

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
                    self.fix_command_btn.config(state=tk.NORMAL)

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
        self.fix_command_btn.config(state=tk.DISABLED)
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
        self.update_status(f"Asking AI for correction for: {failed_cmd}...")
        request_with_error = f"{failed_cmd}ERR_SEPARATOR{error_msg or 'Unknown error'}"
        corrected_commands = self.ai_provider.get_commands(
            request_with_error, manufacturer, self.model_entry.get(), self.ver_entry.get(),
            device_type=device_type, running_config=self.running_config_text.get('1.0', tk.END),
            prompt_style='fix_command'
        )

        # If AI can't find a specific fix, ask for general guidance instead
        if corrected_commands and corrected_commands[0].strip().startswith("# AI Error: Unable to determine correction"):
            self.update_status("No specific fix found, asking for general guidance...")
            guidance_request = f"The command '{failed_cmd}' was incomplete or ambiguous. What are the possible valid commands that could follow it on a {manufacturer} {device_type} device? Provide a brief guide with examples."
            guidance = self.ai_provider.get_commands(
                guidance_request, manufacturer, self.model_entry.get(), self.ver_entry.get(),
                device_type=device_type, running_config=self.running_config_text.get('1.0', tk.END),
                prompt_style='guidance'
            )
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
            self.push_to_device_btn.config(state=tk.NORMAL)
        else:
            self.ai_output.insert(tk.END, "AI Guidance:\r\n")
            self.update_status("AI guidance received.")
            self.push_to_device_btn.config(state=tk.DISABLED)
        
        self.ai_output.insert(tk.END, "------------------------\r\n")
        self.ai_output.insert(tk.END, "\n".join(correction_or_guidance))
        self.fix_command_btn.config(state=tk.DISABLED)

    def run_precheck(self):
        """Run precheck for serial connection - simplified for COM port usage"""
        self.log_to_terminal("\nRunning pre-check to identify device...", "info")
        
        try:
            # Send a simple command to test the connection
            self.connection.write(b'\r\n')
            time.sleep(0.5)
            
            # Try to read any response
            if self.connection.in_waiting > 0:
                response = self.connection.read(self.connection.in_waiting).decode('utf-8', errors='ignore')
                self.log_to_terminal(f"Device response: {response}", "info")
            else:
                self.log_to_terminal("No initial response from device", "info")
                
        except Exception as e:
            self.log_to_terminal(f"Precheck error: {str(e)}", "error")

    def query_ai(self, event=None):
        user_request = self.ai_input.get()
        if not user_request: return
        
        manufacturer = self.man_entry.get()
        if not manufacturer:
            messagebox.showerror("Input Error", "Manufacturer field is required for AI query.")
            return

        self.ai_output.delete('1.0', tk.END)
        self.ai_output.insert(tk.END, f"> User: {user_request}\n\n")
        self.update_status("Querying AI...")
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
        self.update_status("Generating commands via AI…")
        commands = self.ai_provider.get_commands(
            text,
            manufacturer,
            self.model_entry.get(),
            self.ver_entry.get(),
            device_type=self.type_entry.get(),
            running_config=self.running_config_text.get('1.0', tk.END),
            use_web_search=self.use_web_search_var.get(),
            ollama_model=self.ollama_model_combo.get(),
            gemini_model=self.get_selected_gemini_model_full(),
            prompt_style='default'  # First, try to get commands
        )

        # If the first attempt returns an error, automatically switch to guidance mode and retry
        if commands and commands[0].strip().startswith("# AI Error:"):
            self.update_status("Request is complex, asking AI for guidance...")
            self.chat_log.insert(tk.END, f"AI: {commands[0]}\n")
            self.chat_log.insert(tk.END, "\nAI: The request is complex. Here is some general guidance:\n\n")

            guidance_response = self.ai_provider.get_commands(
                text,
                manufacturer,
                self.model_entry.get(),
                self.ver_entry.get(),
                device_type=self.type_entry.get(),
                running_config=self.running_config_text.get('1.0', tk.END),
                use_web_search=self.use_web_search_var.get(),
                ollama_model=self.ollama_model_combo.get(),
                gemini_model=self.get_selected_gemini_model_full(),
                prompt_style='guidance'  # Retry in guidance mode
            )
            # Display guidance and clear the other panes as there are no commands to push
            for line in guidance_response:
                self.chat_log.insert(tk.END, line + "\n")
            self.chat_log.insert(tk.END, "\n")
            self.ai_output.delete('1.0', tk.END)
            self.push_to_device_btn.config(state=tk.DISABLED)
            self.last_chat_response = guidance_response # Save for KB

        else:  # Success on the first try, we have commands
            self.chat_log.insert(tk.END, "AI: Proposed commands:\n")
            for cmd in commands:
                self.chat_log.insert(tk.END, cmd + "\n")
            self.chat_log.insert(tk.END, "\n")
            self.last_chat_response = commands # Save for KB
            # Save successful, non-error commands to the database
            if commands and not any("# AI Error:" in cmd for cmd in commands):
                self._save_commands_to_db(manufacturer, text, commands)

        # Mirror into AI output pane for optional push
        try:
            self.ai_output.delete('1.0', tk.END)
            self.ai_output.insert(tk.END, "AI Generated Commands:\n")
            self.ai_output.insert(tk.END, "------------------------\n")
            self.ai_output.insert(tk.END, "\n".join(commands))
            self.push_to_device_btn.config(state=tk.NORMAL)
        except Exception:
            pass
        self.chat_log.see(tk.END)
        self.update_status("Chat: commands generated")

        self.fix_command_btn.config(state=tk.DISABLED) # Disable after use

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

    def fetch_available_commands(self):
        if not self.connection or not hasattr(self, 'is_connected') or not self.is_connected:
            messagebox.showerror("Error", "Not connected to any device.")
            return

        self.update_status("Fetching available commands with '?'...")
        self.log_to_terminal("\n>>> Fetching available commands with '?'...", "info")
        
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
            
            self._save_knowledge_to_db(self.man_entry.get(), category, commands_output.split('\n'))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch available commands: {e}")
            self.update_status("Failed to fetch available commands.")

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
                    self.man_entry.delete(0, tk.END)
                    self.man_entry.insert(0, vendor_match.group(1).strip())
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
        self.update_status(f"Sending {len(original_commands)} commands...")
        self.update_idletasks()

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
        self.profile_combo['values'] = profile_names
        if profile_names:
            self.profile_combo.set(profile_names)
            self.load_selected_profile()

    def load_selected_profile(self, event=None):
        profile_name = self.profile_combo.get()
        if profile_name in self.profiles:
            profile = self.profiles[profile_name]
            self.com_port_combo.set(profile.get('com_port', ''))
            self.user_entry.delete(0, tk.END); self.user_entry.insert(0, profile.get('username', ''))
            self.pass_entry.delete(0, tk.END); self.pass_entry.insert(0, profile.get('password', ''))
            self.enable_pass_entry.delete(0, tk.END); self.enable_pass_entry.insert(0, profile.get('enable_password', ''))
            self.man_entry.delete(0, tk.END); self.man_entry.insert(0, profile.get('manufacturer', ''))
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
                'available_commands': self.available_commands_text.get('1.0', tk.END)
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
                'available_commands': self.available_commands_text.get('1.0', tk.END)
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
                for entry in [self.user_entry, self.pass_entry, self.man_entry, self.model_entry, self.ver_entry]:
                    entry.delete(0, tk.END)
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
            tree.selection_data[item_id] = row[4] # Store guidance_text

    def _on_kb_select(self, event, tree, text_widget):
        try:
            selected_item = tree.selection()[0]
            guidance = tree.selection_data.get(selected_item, "")
            text_widget.delete('1.0', tk.END)
            text_widget.insert(tk.END, guidance)
        except IndexError:
            pass # Ignore empty selection

if __name__ == "__main__":
    app = NetApp()
    app.mainloop()
