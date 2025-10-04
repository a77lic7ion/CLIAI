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

# --- Manufacturer to Device Category Mapping ---
MANUFACTURER_TO_CATEGORY = {
    "h3c": "Switch",
    "cisco": "Switch",
    "juniper": "Firewall",
    "arista": "Switch",
    "vello": "VELLO",
    # Add more mappings as needed, these are examples
    "fortinet": "Firewall",
    "paloalto": "Firewall",
    "f5": "Firewall",
    "check point": "Firewall",
    "brocade": "Switch",
    "extreme": "Switch",
    "huawei": "Switch",
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

    def get_commands(self, user_request, manufacturer, model, version, use_web_search=False, ollama_model='llama3', gemini_model=None):
        if self.provider == "None": return ["# AI not configured."]
        if self.provider == "Simulation": return self.run_simulation(user_request)

        device_context = f"The target device is a **{manufacturer}**"
        if model: device_context += f" model **{model}**"
        if version: device_context += f" running software version **{version}**"
        device_context += "."

        # --- A much more forceful and explicit system prompt ---
        system_prompt = f"""
        You are a world-class network engineering expert AI. Your one and only task is to generate precise, executable CLI commands for a specific network device.

        **CRITICAL CONTEXT:** The target device is a **{manufacturer.upper()}** device. All commands you generate **MUST** use the correct syntax for **{manufacturer.upper()}**. Do not use syntax from any other vendor, especially Cisco, unless the manufacturer is explicitly set to Cisco.

        **DEVICE DETAILS:** {device_context}

        **YOUR DIRECTIVES:**
        1.  **PRIORITIZE THE MANUFACTURER:** The user-provided manufacturer ({manufacturer.upper()}) is the most important piece of information. Your output must be 100% correct for this vendor.
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
        self.title("Intelligent Network Manager")
        self.geometry("1000x800")

        self.connection = None
        self.serial_queue = queue.Queue()
        self.reader_thread = None
        self.reader_running = False
        self.profiles = {}
        self.profiles_file = 'profiles.json'
        self.ai_provider = AIProvider()
        self.device_commands = {}
        self.commands_file = 'cli_commands.json'

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
        # Populate COM ports now that the combobox exists
        self.refresh_com_ports()
        tk.Label(conn_frame, text="Username:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.user_entry = tk.Entry(conn_frame)
        self.user_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        tk.Label(conn_frame, text="Password:").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.pass_entry = tk.Entry(conn_frame, show="*")
        self.pass_entry.grid(row=2, column=3, padx=5, pady=5, sticky="ew")
        self.connect_btn = tk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        profile_btn_frame = tk.Frame(conn_frame)
        profile_btn_frame.grid(row=0, column=2, columnspan=2, sticky='ew')
        tk.Button(profile_btn_frame, text="Save", command=self.save_profile).pack(side=tk.LEFT, fill='x', expand=True)
        tk.Button(profile_btn_frame, text="Delete", command=self.delete_profile).pack(side=tk.LEFT, fill='x', expand=True)

        terminal_frame = tk.LabelFrame(left_frame, text="Device Terminal", bd=2, relief=tk.GROOVE)
        terminal_frame.pack(pady=10, padx=10, expand=True, fill="both")
        self.terminal = scrolledtext.ScrolledText(terminal_frame, wrap=tk.WORD, bg="black", fg="white", font=("Consolas", 10))
        self.terminal.pack(expand=True, fill="both")
        # Simple terminal input for interactive CLI
        term_input_frame = tk.Frame(terminal_frame)
        term_input_frame.pack(fill="x", padx=10, pady=5)
        self.term_input = tk.Entry(term_input_frame, font=("Consolas", 10))
        self.term_input.pack(side=tk.LEFT, fill="x", expand=True)
        self.term_input.bind("<Return>", self.send_terminal_input)
        tk.Button(term_input_frame, text="Send", command=self.send_terminal_input).pack(side=tk.LEFT, padx=5)

        # Terminal options: wrap mode and clear
        terminal_opts_frame = tk.Frame(terminal_frame)
        terminal_opts_frame.pack(fill="x", padx=10, pady=2)
        tk.Label(terminal_opts_frame, text="Terminal Wrap:").pack(side=tk.LEFT)
        self.term_wrap_combo = ttk.Combobox(terminal_opts_frame, state="readonly", values=["Wrap (word)", "No wrap"], width=12)
        self.term_wrap_combo.set("Wrap (word)")
        self.term_wrap_combo.bind("<<ComboboxSelected>>", self.on_term_wrap_change)
        self.term_wrap_combo.pack(side=tk.LEFT, padx=5)
        tk.Button(terminal_opts_frame, text="Font +", command=self.increase_terminal_font).pack(side=tk.LEFT, padx=5)
        tk.Button(terminal_opts_frame, text="Font -", command=self.decrease_terminal_font).pack(side=tk.LEFT)
        # Pager controls
        self.auto_pager_var = tk.BooleanVar(value=True)
        tk.Checkbutton(terminal_opts_frame, text="Auto-advance pager", variable=self.auto_pager_var).pack(side=tk.LEFT, padx=10)
        tk.Button(terminal_opts_frame, text="Next Page", command=self.send_pager_next).pack(side=tk.LEFT)
        tk.Button(terminal_opts_frame, text="Stop Paging", command=self.send_pager_stop).pack(side=tk.LEFT, padx=5)
        tk.Button(terminal_opts_frame, text="Export…", command=self.export_terminal_chat).pack(side=tk.RIGHT, padx=5)
        tk.Button(terminal_opts_frame, text="Clear", command=self.clear_terminal).pack(side=tk.RIGHT)

        ai_pane = tk.PanedWindow(main_frame, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        main_pane.add(ai_pane)

        ai_config_frame = tk.LabelFrame(ai_pane, text="AI Configuration", relief=tk.GROOVE)
        ai_pane.add(ai_config_frame, height=150)
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
        # Apply requested defaults and initialize provider
        try:
            self.api_key_entry.delete(0, tk.END)
            self.api_key_entry.insert(0, "AIzaSyASna_AVZT5NynbpU1eNzXwnxWgo4f_6Lc")
            self.set_ai_provider()
        except Exception:
            pass

        ai_assistant_frame = tk.LabelFrame(ai_pane, text="AI Assistant", relief=tk.GROOVE)
        ai_pane.add(ai_assistant_frame)
        context_frame = tk.Frame(ai_assistant_frame)
        context_frame.pack(pady=5, padx=10, fill='x')
        tk.Label(context_frame, text="Manufacturer:").grid(row=0, column=0, sticky="w")
        self.man_entry = tk.Entry(context_frame)
        self.man_entry.grid(row=0, column=1, sticky="ew", padx=2)
        tk.Label(context_frame, text="Model:").grid(row=1, column=0, sticky="w")
        self.model_entry = tk.Entry(context_frame)
        self.model_entry.grid(row=1, column=1, sticky="ew", padx=2)
        tk.Label(context_frame, text="Version:").grid(row=2, column=0, sticky="w")
        self.ver_entry = tk.Entry(context_frame)
        self.ver_entry.grid(row=2, column=1, sticky="ew", padx=2)
        tk.Label(context_frame, text="Type:").grid(row=3, column=0, sticky="w")
        self.type_entry = tk.Entry(context_frame, state='readonly')
        self.type_entry.grid(row=3, column=1, sticky="ew", padx=2)

        self.fetch_info_btn = tk.Button(context_frame, text="Fetch Device Info", command=self.fetch_device_info)
        self.fetch_info_btn.grid(row=4, column=0, columnspan=2, pady=5, sticky="ew")

        context_frame.columnconfigure(1, weight=1)
        ttk.Separator(ai_assistant_frame, orient='horizontal').pack(fill='x', pady=5, padx=10)
        tk.Label(ai_assistant_frame, text="Your Request:").pack(pady=5, padx=10, anchor="w")
        self.ai_input = tk.Entry(ai_assistant_frame, font=("Arial", 10))
        self.ai_input.pack(pady=5, padx=10, fill="x")
        self.ai_input.bind("<Return>", self.query_ai)
        self.use_web_search_var = tk.BooleanVar(value=True)
        self.web_search_check = tk.Checkbutton(ai_assistant_frame, text="Use Web Search (Gemini)", variable=self.use_web_search_var)
        self.web_search_check.pack(pady=5, padx=10, anchor="w")

        ai_btn_frame = tk.Frame(ai_assistant_frame)
        ai_btn_frame.pack(pady=5, padx=10, fill="x")
        tk.Button(ai_btn_frame, text="Generate Commands", command=self.query_ai).pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        tk.Button(ai_btn_frame, text="Fetch Recommended Commands", command=self.fetch_recommended_commands).pack(side=tk.LEFT, fill='x', expand=True, padx=(5, 0))

        self.ai_output = scrolledtext.ScrolledText(ai_assistant_frame, wrap=tk.WORD, height=10, font=("Consolas", 10))
        self.ai_output.pack(pady=10, padx=10, expand=True, fill="both")

        action_btn_frame = tk.Frame(ai_assistant_frame)
        action_btn_frame.pack(pady=10, padx=10, fill="x")
        self.push_to_device_btn = tk.Button(action_btn_frame, text="Push to Connected Device", command=self.push_ai_commands, state=tk.DISABLED)
        self.push_to_device_btn.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        tk.Button(action_btn_frame, text="Save Commands", command=self.save_fetched_commands).pack(side=tk.LEFT, fill='x', expand=True, padx=(5, 0))

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.load_profiles()
        self.load_device_commands()
        self.on_ai_provider_change()
        self.update_status("Ready")

    def load_device_commands(self):
        if os.path.exists(self.commands_file):
            with open(self.commands_file, 'r') as f:
                try:
                    self.device_commands = json.load(f)
                    self.log_to_terminal(f"Loaded device commands from {self.commands_file}", "info")
                except json.JSONDecodeError:
                    self.log_to_terminal(f"Warning: Could not decode {self.commands_file}. Starting fresh.", "error")
                    self.device_commands = {}
        else:
            self.log_to_terminal("No device command file found. A new one will be created.", "info")
            self.device_commands = {}

    def save_device_commands(self):
        try:
            with open(self.commands_file, 'w') as f:
                json.dump(self.device_commands, f, indent=4)
            self.log_to_terminal(f"Saved device commands to {self.commands_file}", "info")
        except Exception as e:
            self.log_to_terminal(f"Error saving commands: {e}", "error")

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

    def connect(self):
        com_port = self.com_port_combo.get()
        username = self.user_entry.get()
        password = self.pass_entry.get()
        
        if not com_port:
            messagebox.showerror("Error", "Please select a COM port")
            return
        
        try:
            # Create serial connection
            self.connection = serial.Serial(
                port=com_port,
                baudrate=9600,  # Default baudrate, can be made configurable
                timeout=0,      # Non-blocking reads
                write_timeout=1
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
            self._start_serial_reader()
            
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect to {com_port}: {str(e)}")

    def disconnect(self):
        if self.connection:
            try:
                self._stop_serial_reader()
                self.connection.close()
                self.connection = None
                self.is_connected = False
                self.connect_btn.config(text="Connect")
                self.push_to_device_btn.config(state=tk.DISABLED)
                self.log_to_terminal("Disconnected")
                self.update_status("Disconnected")
            except Exception as e:
                messagebox.showerror("Disconnection Error", f"Error during disconnection: {str(e)}")

    def _start_serial_reader(self):
        if self.reader_thread and self.reader_thread.is_alive():
            return
        self.reader_running = True
        self.reader_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
        self.reader_thread.start()
        # Start draining queue on the Tk mainloop
        self.after(50, self._drain_serial_queue)

    def _stop_serial_reader(self):
        self.reader_running = False
        # Thread is daemon; it will exit automatically. Queue draining will stop when no data.

    def _serial_reader_loop(self):
        while self.reader_running and self.connection:
            try:
                # Read any available bytes
                waiting = self.connection.in_waiting
                if waiting:
                    data = self.connection.read(waiting)
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
        except queue.Empty:
            pass
        if self.reader_running:
            # Continue polling while reader active
            self.after(50 if drained else 100, self._drain_serial_queue)

    def send_terminal_input(self, event=None):
        cmd = self.term_input.get().strip()
        # Local CLI: clear terminal regardless of connection
        if cmd.lower() in ("clear", "cls"):
            self.clear_terminal()
            self.term_input.delete(0, tk.END)
            return
        if not cmd:
            return
        if not self.connection or not getattr(self, "is_connected", False):
            messagebox.showerror("Error", "Not connected to any device")
            return
        try:
            # Echo command to terminal and send to device
            self.log_to_terminal(f"\n> {cmd}\r\n", "command")
            self.connection.write((cmd + "\r\n").encode())
            self.term_input.delete(0, tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send: {e}")
            self.log_to_terminal(f"\n[Send error] {e}\n", "error")

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
            self.use_web_search_var.get(),
            self.ollama_model_combo.get(),
            self.get_selected_gemini_model_full()
        )
        
        self.ai_output.insert(tk.END, "AI Generated Commands:\n")
        self.ai_output.insert(tk.END, "------------------------\n")
        self.ai_output.insert(tk.END, "\n".join(commands))
        self.update_status("AI response received.")

    def fetch_recommended_commands(self):
        manufacturer = self.man_entry.get()
        device_type = self.type_entry.get()
        model = self.model_entry.get()

        if not manufacturer or not device_type:
            messagebox.showerror("Input Error", "Manufacturer and Type are required to fetch recommendations.")
            return

        prompt = f"List the most common and useful operational CLI commands for a {manufacturer} {model} {device_type}. Provide only the commands, one per line, without any explanation or markdown."

        self.ai_output.delete('1.0', tk.END)
        self.ai_output.insert(tk.END, f"> Fetching recommended commands for: {manufacturer} {device_type}\n\n")
        self.update_status("Querying AI for recommended commands...")
        self.update_idletasks()

        commands = self.ai_provider.get_commands(
            prompt,
            manufacturer,
            model,
            self.ver_entry.get(),
            self.use_web_search_var.get(),
            self.ollama_model_combo.get(),
            self.get_selected_gemini_model_full()
        )

        self.ai_output.insert(tk.END, "AI Generated Commands:\n")
        self.ai_output.insert(tk.END, "------------------------\n")
        self.ai_output.insert(tk.END, "\n".join(commands))
        self.update_status("AI response received.")

    def fetch_device_info(self):
        if not self.connection or not hasattr(self, 'is_connected') or not self.is_connected:
            messagebox.showerror("Error", "Not connected to any device.")
            return

        manufacturer = self.man_entry.get().lower().strip()
        if not manufacturer:
            messagebox.showerror("Input Error", "Manufacturer is required to fetch device info.")
            return

        # Determine the correct command
        if manufacturer == 'h3c':
            cmd = 'display version'
        else:
            # A common default for many vendors like Cisco, Juniper, Arista
            cmd = 'show version'

        self.log_to_terminal(f"\n>>> Fetching device info with '{cmd}'...", "info")
        # Track attempts to allow fallback commands
        self.device_info_attempt = 1

        # Clear old info
        self.model_entry.delete(0, tk.END)
        self.ver_entry.delete(0, tk.END)

        # Send the command
        try:
            # The response will be read by the async serial reader and appended to the terminal.
            # We schedule a parsing function to run after a delay to capture the output.
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

        # Determine and set device type
        device_type = MANUFACTURER_TO_CATEGORY.get(manufacturer.lower(), "Unknown")
        self.type_entry.config(state='normal')
        self.type_entry.delete(0, tk.END)
        self.type_entry.insert(0, device_type)
        self.type_entry.config(state='readonly')
        self.log_to_terminal(f"Determined Device Type: {device_type}", "info")

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
        lines = commands_text.split("------------------------\n")
        if len(lines) < 2:
            messagebox.showinfo("Info", "No commands to push.")
            return
        
        commands_to_push = [cmd.strip() for cmd in lines[1].strip().split("\n") if cmd.strip() and not cmd.startswith("#")]
        
        if not commands_to_push:
            messagebox.showinfo("Info", "No valid commands found to push.")
            return

        self.log_to_terminal(f"\n>>> Pushing {len(commands_to_push)} commands from AI Assistant...", "info")
        self.update_status(f"Sending {len(commands_to_push)} commands...")
        self.update_idletasks()
        
        try:
            for cmd in commands_to_push:
                self.log_to_terminal(f"\n> {cmd}", "command")
                # Send command via serial connection
                self.connection.write((cmd + '\r\n').encode())
                
                # Read response (with timeout)
                response = ""
                start_time = time.time()
                while time.time() - start_time < 3:  # 3 second timeout per command
                    if self.connection.in_waiting > 0:
                        response += self.connection.read(self.connection.in_waiting).decode('utf-8', errors='ignore')
                    time.sleep(0.1)
                
                if response:
                    self.log_to_terminal(response, "output")
                else:
                    self.log_to_terminal("No response received", "info")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send commands: {str(e)}")
            self.log_to_terminal(f"Error sending commands: {str(e)}", "error")
            
        self.update_status("Commands sent. Ready.")

    def save_fetched_commands(self):
        manufacturer = self.man_entry.get().lower()
        device_type = self.type_entry.get()

        if not manufacturer or not device_type:
            messagebox.showerror("Error", "Manufacturer and Type are required to save commands.")
            return

        commands_text = self.ai_output.get("1.0", tk.END)
        lines = commands_text.split("------------------------\n")
        if len(lines) < 2:
            messagebox.showinfo("Info", "No commands to save.")
            return

        commands_to_save = [cmd.strip() for cmd in lines[1].strip().split("\n") if cmd.strip() and not cmd.startswith("#")]

        if not commands_to_save:
            messagebox.showinfo("Info", "No valid commands found to save.")
            return

        # Ensure manufacturer and device_type keys exist
        if manufacturer not in self.device_commands:
            self.device_commands[manufacturer] = {}
        if device_type not in self.device_commands[manufacturer]:
            self.device_commands[manufacturer][device_type] = []

        # Add only new commands to avoid duplicates
        existing_commands = self.device_commands[manufacturer][device_type]
        newly_added_count = 0
        for cmd in commands_to_save:
            if cmd not in existing_commands:
                existing_commands.append(cmd)
                newly_added_count += 1

        if newly_added_count > 0:
            self.save_device_commands()
            messagebox.showinfo("Success", f"{newly_added_count} new command(s) saved for {manufacturer} - {device_type}.")
        else:
            messagebox.showinfo("Info", "All commands are already saved for this device type.")

    def log_to_terminal(self, message, tag=None):
        self.terminal.tag_config("info", foreground="cyan")
        self.terminal.tag_config("error", foreground="red")
        self.terminal.tag_config("command", foreground="yellow")
        self.terminal.tag_config("prompt", foreground="lime green")
        
        self.terminal.insert(tk.END, message + "\n", tag)
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
            with open(self.profiles_file, 'r') as f:
                self.profiles = json.load(f)
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
            self.man_entry.delete(0, tk.END); self.man_entry.insert(0, profile.get('manufacturer', ''))
            self.model_entry.delete(0, tk.END); self.model_entry.insert(0, profile.get('model', ''))
            self.ver_entry.delete(0, tk.END); self.ver_entry.insert(0, profile.get('version', ''))
            self.type_entry.config(state='normal')
            self.type_entry.delete(0, tk.END); self.type_entry.insert(0, profile.get('device_type', ''))
            self.type_entry.config(state='readonly')
            self.update_status(f"Loaded profile: {profile_name}")

    def save_profile(self):
        profile_name = simpledialog.askstring("Save Profile", "Enter a name for this profile:")
        if profile_name:
            self.profiles[profile_name] = {
                'com_port': self.com_port_combo.get(),
                'username': self.user_entry.get(),
                'password': self.pass_entry.get(),
                'manufacturer': self.man_entry.get(),
                'model': self.model_entry.get(),
                'version': self.ver_entry.get(),
                'device_type': self.type_entry.get()
            }
            with open(self.profiles_file, 'w') as f:
                json.dump(self.profiles, f, indent=4)
            self.update_profile_list()
            self.profile_combo.set(profile_name)
            self.update_status(f"Saved profile: {profile_name}")

    def delete_profile(self):
        profile_name = self.profile_combo.get()
        if not profile_name: return
        if messagebox.askyesno("Confirm", f"Are you sure you want to delete the profile '{profile_name}'?"):
            if profile_name in self.profiles:
                del self.profiles[profile_name]
                with open(self.profiles_file, 'w') as f: json.dump(self.profiles, f, indent=4)
                self.com_port_combo.set('')
                for entry in [self.user_entry, self.pass_entry, self.man_entry, self.model_entry, self.ver_entry, self.type_entry]:
                    if entry == self.type_entry:
                        entry.config(state='normal')
                        entry.delete(0, tk.END)
                        entry.config(state='readonly')
                    else:
                        entry.delete(0, tk.END)
                self.update_profile_list()
                self.update_status(f"Deleted profile: {profile_name}")

if __name__ == "__main__":
    app = NetApp()
    app.mainloop()
