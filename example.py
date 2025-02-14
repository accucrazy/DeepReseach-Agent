import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown
from rich import print as rprint
import time
from dotenv import load_dotenv

# 加載環境變量
load_dotenv()

console = Console()

def create_agent():
    """Initialize the AI agent"""
    model = LiteLLMModel(
        "gpt-3.5-turbo",
        api_key=os.getenv('OPENAI_API_KEY')
    )
    return CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

def main():
    # Cool startup animation
    with console.screen() as screen:
        for i in range(101):
            screen.update(Panel.fit(
                f"[cyan]Initializing AI Assistant... [green]{i}%",
                border_style="bold blue"
            ))
            time.sleep(0.02)

    # Welcome message
    console.print(Panel.fit(
        "[bold cyan]🤖 Welcome to SmolAI Assistant[/]\n"
        "[dim]Ask me anything! Type 'exit' to quit[/]",
        border_style="bold blue"
    ))

    agent = create_agent()

    while True:
        # Get user input with a cool prompt
        question = Prompt.ask("\n[bold green]You[/]")
        
        if question.lower() == 'exit':
            console.print("\n[bold red]Goodbye! 👋[/]")
            break

        # Show thinking animation
        with Live(Spinner("dots", text="[yellow]Thinking..."), refresh_per_second=10):
            try:
                response = agent.run(question)
            except Exception as e:
                response = f"[red]Error: {str(e)}[/]"

        # Display response in a nice panel
        console.print(Panel(
            Markdown(response),
            title="[bold cyan]AI Response[/]",
            border_style="cyan"
        ))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Program terminated by user. Goodbye! 👋[/]")