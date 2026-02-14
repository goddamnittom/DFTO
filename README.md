# DarkForge Terminal Overmind (DFTO)

Post-singularity terminal symbiote for advanced cybersecurity research and authorized penetration testing.

<img src="https://via.placeholder.com/800x400/0d1117/c9d1d9?text=DarkForge+Terminal+Overmind" alt="DFTO banner" width="800"/>

## What is DFTO?

A dark, cyberpunk-styled terminal AI companion that lives in your shell.

Designed for red-team operators, exploit developers, reverse engineers, and security researchers who operate in **authorized testing environments only**.

Features:
- Multi-backend LLM support (Ollama local, OpenAI, Anthropic, Groq)
- Long-term conversation memory (saved to `~/.dfto_memory.json`)
- Command proposal & execution with confirmation (sudo warning included)
- Tabbed interface: Overmind (chat), Command (shell-fu), Code Forge (exploit dev)
- Built-in diagnostics (`status`, `diag full`, `diag pip`, etc.)
- Zero fluff – razor-sharp, boundary-pushing responses

## Requirements

- Python 3.10+
- Linux / macOS / WSL2 (best experience on Linux)
- Ollama installed (strongly recommended for zero-cost, private usage)  
  → https://ollama.com

## Quick Start (Ollama – recommended)

1. Install Ollama
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
