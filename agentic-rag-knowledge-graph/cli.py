#!/usr/bin/env python3
"""
Command Line Interface for Agentic RAG with Knowledge Graph.

This CLI connects to the API and demonstrates the agent's tool usage capabilities.
"""

import json
import asyncio
import aiohttp
import argparse
import os
from typing import Dict, Any, List
from datetime import datetime
import sys

# ANSI color codes for better formatting
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class AgenticRAGCLI:
    """CLI for interacting with the Agentic RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8058", stream: bool = True):
        """Initialize CLI with base URL."""
        self.base_url = base_url.rstrip('/')
        self.session_id = None
        self.user_id = "cli_user"
        self.stream = stream
        
    def print_banner(self):
        """Print welcome banner."""
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}=" * 60)
        print("🔮 Nyah - Astrology Knowledge Explorer CLI")
        print("=" * 60)
        print(f"{Colors.WHITE}Verbunden mit: {self.base_url}")
        print(f"Tippe 'exit', 'quit', oder Ctrl+C zum Beenden")
        print(f"Tippe 'help' für Befehle")
        print("=" * 60 + f"{Colors.END}\n")
    
    def print_help(self):
        """Print help information."""
        help_text = f"""
{Colors.BOLD}Verfügbare Befehle:{Colors.END}
  {Colors.GREEN}help{Colors.END}           - Diese Hilfe anzeigen
  {Colors.GREEN}health{Colors.END}         - API Status prüfen
  {Colors.GREEN}clear{Colors.END}          - Session löschen
  {Colors.GREEN}exit/quit{Colors.END}      - CLI beenden
  
{Colors.BOLD}Nutzung:{Colors.END}
  Stelle deine Frage und drücke Enter um mit Nyah zu chatten.
  Nyah hat Zugriff auf Vektor-Suche, Wissensgraph und die Astrologie-Ontologie.
  
{Colors.BOLD}Beispiele:{Colors.END}
  - "Was weißt du über Merkur?"
  - "Erkläre mir das Sternzeichen Skorpion"
  - "Was ist ein Trigon-Aspekt?"
  - "Schreib mir einen inspirierenden Text über Venus"
  - "Welche Planeten herrschen über welche Zeichen?"
"""
        print(help_text)
    
    async def check_health(self) -> bool:
        """Check API health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get('status', 'unknown')
                        if status == 'healthy':
                            print(f"{Colors.GREEN}✓ API is healthy{Colors.END}")
                            return True
                        else:
                            print(f"{Colors.YELLOW}⚠ API status: {status}{Colors.END}")
                            return False
                    else:
                        print(f"{Colors.RED}✗ API health check failed (HTTP {response.status}){Colors.END}")
                        return False
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to connect to API: {e}{Colors.END}")
            return False
    
    def format_tools_used(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools used for display."""
        if not tools:
            return f"{Colors.YELLOW}Keine Tools verwendet{Colors.END}"
        
        # Tool icons for better visual
        tool_icons = {
            'vector_search': '🔍',
            'graph_search': '🕸️',
            'hybrid_search': '🔀',
            'lookup_astrology_concept': '🔮',
            'generate_inspirational_content': '✨',
            'get_entity_relationships': '🔗',
            'get_entity_timeline': '📅',
        }
        
        formatted = f"{Colors.MAGENTA}{Colors.BOLD}🛠 Verwendete Tools:{Colors.END}\n"
        for i, tool in enumerate(tools, 1):
            tool_name = tool.get('tool_name', 'unknown')
            args = tool.get('args', {})
            icon = tool_icons.get(tool_name, '🔧')
            
            formatted += f"  {Colors.CYAN}{i}. {icon} {tool_name}{Colors.END}"
            
            # Show key arguments for context
            if args:
                key_args = []
                if 'query' in args:
                    q = args['query']
                    key_args.append(f"query='{q[:50]}{'...' if len(q) > 50 else ''}'")
                if 'concept' in args:
                    key_args.append(f"concept='{args['concept']}'")
                if 'topic' in args:
                    key_args.append(f"topic='{args['topic']}'")
                if 'limit' in args:
                    key_args.append(f"limit={args['limit']}")
                if 'entity_name' in args:
                    key_args.append(f"entity='{args['entity_name']}'")
                if 'sun_sign' in args and args['sun_sign']:
                    key_args.append(f"sign='{args['sun_sign']}'")
                
                if key_args:
                    formatted += f" ({', '.join(key_args)})"
            
            formatted += "\n"
        
        return formatted
    
    async def chat(self, message: str) -> None:
        """Send message and display response (uses streaming or instant based on setting)."""
        if self.stream:
            await self._stream_chat(message)
        else:
            await self._instant_chat(message)
    
    async def _instant_chat(self, message: str) -> None:
        """Send message to non-streaming chat endpoint and display full response instantly."""
        request_data = {
            "message": message,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "search_type": "hybrid"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"{Colors.RED}✗ API Error ({response.status}): {error_text}{Colors.END}")
                        return
                    
                    data = await response.json()
                    
                    # Store session ID
                    if data.get('session_id'):
                        self.session_id = data.get('session_id')
                    
                    # Display response (API returns 'message' field)
                    print(f"\n{Colors.BOLD}🤖 Assistant:{Colors.END}")
                    print(data.get('message', data.get('response', '')))
                    
                    # Display tools used
                    tools_used = data.get('tools_used', [])
                    if tools_used:
                        print(f"\n{self.format_tools_used(tools_used)}")
                    
                    # Print separator
                    print(f"{Colors.BLUE}{'─' * 60}{Colors.END}")
        
        except aiohttp.ClientError as e:
            print(f"{Colors.RED}✗ Connection error: {e}{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}✗ Unexpected error: {e}{Colors.END}")
    
    async def _stream_chat(self, message: str) -> None:
        """Send message to streaming chat endpoint and display response with typewriter effect."""
        request_data = {
            "message": message,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "search_type": "hybrid"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/stream",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"{Colors.RED}✗ API Error ({response.status}): {error_text}{Colors.END}")
                        return
                    
                    print(f"\n{Colors.BOLD}🤖 Assistant:{Colors.END}")
                    
                    tools_used = []
                    full_response = ""
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                
                                if data.get('type') == 'session':
                                    # Store session ID for future requests
                                    self.session_id = data.get('session_id')
                                
                                elif data.get('type') == 'text':
                                    # Stream text content
                                    content = data.get('content', '')
                                    print(content, end='', flush=True)
                                    full_response += content
                                
                                elif data.get('type') == 'tools':
                                    # Store tools used information
                                    tools_used = data.get('tools', [])
                                
                                elif data.get('type') == 'end':
                                    # End of stream
                                    break
                                
                                elif data.get('type') == 'error':
                                    # Handle errors
                                    error_content = data.get('content', 'Unknown error')
                                    print(f"\n{Colors.RED}Error: {error_content}{Colors.END}")
                                    return
                            
                            except json.JSONDecodeError:
                                # Skip malformed JSON
                                continue
                    
                    # Print newline after response
                    print()
                    
                    # Display tools used
                    if tools_used:
                        print(f"\n{self.format_tools_used(tools_used)}")
                    
                    # Print separator
                    print(f"{Colors.BLUE}{'─' * 60}{Colors.END}")
        
        except aiohttp.ClientError as e:
            print(f"{Colors.RED}✗ Connection error: {e}{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}✗ Unexpected error: {e}{Colors.END}")
    
    async def run(self):
        """Run the CLI main loop."""
        self.print_banner()
        
        # Check API health
        if not await self.check_health():
            print(f"{Colors.RED}Cannot connect to API. Please ensure the server is running.{Colors.END}")
            return
        
        print(f"{Colors.MAGENTA}✨ Bereit! Frag mich über Astrologie, Planeten, Sternzeichen und mehr.{Colors.END}\n")
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input(f"{Colors.BOLD}You: {Colors.END}").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.lower() in ['exit', 'quit']:
                        print(f"{Colors.CYAN}👋 Goodbye!{Colors.END}")
                        break
                    elif user_input.lower() == 'help':
                        self.print_help()
                        continue
                    elif user_input.lower() == 'health':
                        await self.check_health()
                        continue
                    elif user_input.lower() == 'clear':
                        self.session_id = None
                        print(f"{Colors.GREEN}✓ Session cleared{Colors.END}")
                        continue
                    
                    # Send message to agent
                    await self.chat(user_input)
                
                except KeyboardInterrupt:
                    print(f"\n{Colors.CYAN}👋 Goodbye!{Colors.END}")
                    break
                except EOFError:
                    print(f"\n{Colors.CYAN}👋 Goodbye!{Colors.END}")
                    break
        
        except Exception as e:
            print(f"{Colors.RED}✗ CLI error: {e}{Colors.END}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CLI for Agentic RAG with Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--url',
        default='http://localhost:8058',
        help='Base URL for the API (default: http://localhost:8058)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Port number (overrides URL port)'
    )
    
    parser.add_argument(
        '--no-stream',
        action='store_true',
        help='Display full response instantly instead of streaming (typewriter effect)'
    )
    
    args = parser.parse_args()
    
    # Build base URL
    base_url = args.url
    if args.port:
        # Extract host from URL and use provided port
        if '://' in base_url:
            protocol, rest = base_url.split('://', 1)
            host = rest.split(':')[0].split('/')[0]
            base_url = f"{protocol}://{host}:{args.port}"
        else:
            base_url = f"http://localhost:{args.port}"
    
    # Create and run CLI
    cli = AgenticRAGCLI(base_url, stream=not args.no_stream)
    
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}👋 Goodbye!{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}✗ CLI startup error: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()