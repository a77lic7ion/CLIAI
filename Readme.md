NetIntelli X
🚀 Project Overview
NetIntelli X is a powerful Python-based desktop application designed to revolutionize network device configuration and management. It integrates robust network automation capabilities with advanced Artificial Intelligence (AI) to streamline operations, reduce errors, and provide context-aware assistance to network engineers.
Leveraging Netmiko for secure SSH connectivity and popular Large Language Models (LLMs) like Google Gemini, OpenAI, and local Ollama models, this application allows users to generate and deploy vendor-speciﬁc CLI commands using natural language requests. It enhances eﬃciency by understanding the live running conﬁguration of devices, ensuring generated commands are accurate and contextually relevant.

✨ Key Features
Multi-Vendor Network Connectivity: Secure SSH connections to H3C, Cisco, Juniper, Arista, and other devices via Netmiko.
AI-Powered Command Generation: Translate natural language requests into precise, vendor-speciﬁc CLI commands using Gemini, OpenAI, or Ollama.
Context-Aware AI Assistance: AI considers device manufacturer, model, version, and the live running conﬁguration for highly accurate command suggestions.
Automated Device Discovery: Pre-checks upon connection automatically identify and populate device model and software version.
Live Conﬁguration Fetch: Fetch and display the current running conﬁguration from connected devices, providing critical context to the AI.
Direct Command Deployment: Push AI-generated commands directly to the connected device with a single click.
Connection Proﬁle Management: Save and load device connection details and AI context for quick access.
Interactive Terminal: Real-time display of device output and manual command input.
Flexible AI Backend: Choose between cloud-based LLMs (Gemini, OpenAI) or local models (Ollama).

Intuitive GUI: User-friendly interface built with Tkinter for ease of use.

📸 Screenshots
(Placeholder for future screenshots. You can add images here once the application is running.)

⚙ Installation
To get started with NetIntelli X, follow these steps:

Clone the Repository (or download the ﬁles)



Alternatively, you can download the directly into a single directory.
,
, and
ﬁles

Python Environment
Ensure you have Python 3.x installed on your system. It is recommended to use a virtual environment to manage dependencies.

Install Dependencies
Install the required Python libraries using	:

Ollama Setup (Optional, for local LLMs)
If you plan to use Ollama for local AI models:

Download and Install Ollama: Follow the instructions on the oﬃcial Ollama website: https://ollama.com/download

Download Models: Pull the desired models (e.g., For example:
Ensure the Ollama server is running on
,

.
) using the Ollama CLI.

🚀 Usage
To run the application, navigate to the project directory in your terminal and execute:

Application Walkthrough
Connection Panel:
Proﬁle Management: Use the Proﬁle  dropdown to load saved connection details.
You can	new proﬁles or Delete existing ones.

Device Details: Enter the
,
, and
for your network device.

Manufacturer: Crucially, enter the device manufacturer (e.g.,
,
,	,

). This informs Netmiko and the AI about the correct syntax to use.

Connect/Disconnect: Click
to establish an SSH session. The button will

change to
Device Terminal:
when connected.

This black-background window displays all communication with the connected device, including commands sent, output received, and application logs.
AI Assistant Tab:
AI Conﬁguration:

Provider: Select your preferred AI model (
).
,
,
, or

API Key: Enter your API key for cloud-based providers (Gemini, OpenAI). This ﬁeld will be disabled for Ollama and Simulation.

Ollama Model: If
is selected, this dropdown will populate with models

available on your local Ollama server.

Click
to apply your AI settings.

AI Interaction:
Manufacturer, Model, Version: These ﬁelds are automatically populated after a

successful connection via a context to the AI.
pre-check. They provide critical

Your Request: Type your natural language request (e.g., "create a default gateway of 10.1.0.1 and dns server of 8.8.8.8 and trunk all ports from Gi1/0/1 - Gi1/0/24").
Use Web Search (Gemini): Check this box to enable web search for Gemini, potentially improving accuracy.

Click
The
Click
network device.
Device Conﬁg Tab:
to get CLI commands from the AI. area will display the AI's output.
to send the generated commands to your live

Fetch Running Conﬁg: Click this button (available when connected) to retrieve the live running conﬁguration from your device. This conﬁguration is then used by the AI as context for subsequent requests.
The fetched conﬁguration will be displayed in the text area below.
Status Bar:
Located at the bottom of the window, it provides real-time feedback on the application's operations and status.

🔧 Conﬁguration

Connection proﬁles are stored in a	ﬁle in the same directory as	.
This ﬁle is automatically created and managed by the application. It stores sensitive information like passwords in plain text. Ensure this ﬁle is secured appropriately on your local machine.

AI API Keys
API keys for Gemini and OpenAI are entered directly into the GUI. These are not persistently stored by the application beyond the current session for security reasons. You will need to re-enter them if you restart the application or switch providers.

⚠ Troubleshooting
ModuleNotFoundError: No module named 'telnetlib' : This indicates you are likely running
Python 3.13 or newer, which removed telnetlib . Ensure your	version is up-to-
date ( pip install --upgrade netmiko ) or use a Python version older than 3.13.
Failed to connect to Ollama : Verify that the Ollama server is running on http://localhost:11434 and that you have downloaded the models you intend to use (e.g., ollama pull llama3 ). Check your ﬁrewall settings.

AI generates incorrect vendor commands: Ensure the
ﬁeld in the

Connection panel is correctly ﬁlled. The AI uses this crucial context to generate vendor- speciﬁc CLI. If the issue persists, try making your natural language request more explicit (e.g., "for an H3C switch, create...").
GUI is unresponsive: Long-running network operations can sometimes cause the GUI to freeze. This application uses a basic Tkinter loop. For very long operations, consider running them in a separate thread (advanced usage).
	AttributeError: '_tkinter.tkapp' object has no attribute 'update_status' : This indicates a missing update_status method or status bar initialization. Ensure you are using the latest app_gui.py provided, which includes the status bar implementation.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, bug ﬁxes, or new features, please feel free to:
Fork the repository.
