from datetime import datetime
import os
from typing import Dict, List, Optional, Any
import json
from enum import Enum
from pydantic import BaseModel, Field, field_validator

from dotenv import load_dotenv
import torch
from smolagents import (
    Tool,
    CodeAgent,
    LiteLLMModel,
    CODE_SYSTEM_PROMPT,
    DuckDuckGoSearchTool,
)
from duckduckgo_search import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install
from antaryaami.ai_flow.smol_models import RAGState
from antaryaami.ai_flow.smol_prompts import PREDICTION_AGENT_PROMPT, AGENT_TASK_PROMPT

# Install rich traceback handler
install(show_locals=True)

# Initialize logging
console = Console()


class LogLevel(str, Enum):
    """Log levels for the application"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogMessage(BaseModel):
    """Structure for log messages"""

    timestamp: datetime = Field(default_factory=datetime.now)
    level: LogLevel
    step: str
    message: str
    error: Optional[Exception] = None
    context: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


def log_event(
    level: LogLevel,
    step: str,
    message: str,
    error: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Unified logging function"""
    log_entry = LogMessage(
        level=level, step=step, message=message, error=error, context=context
    )

    # Format the message
    msg = f"[{log_entry.level}] {log_entry.step}: {log_entry.message}"

    # Add context if available
    if context:
        msg += f"\nContext: {context}"

    # Add error details if available
    if error:
        msg += f"\nError: {str(error)}"
        if hasattr(error, "__traceback__"):
            console.print_exception()

    # Print with appropriate styling
    style = {
        LogLevel.DEBUG: "dim",
        LogLevel.INFO: "white",
        LogLevel.WARNING: "yellow",
        LogLevel.ERROR: "red",
        LogLevel.CRITICAL: "bold red",
    }[level]

    console.print(Panel(msg, style=style))


class DeviceConfig(BaseModel):
    """Configuration for device and batch size"""

    device: str = Field(default="cpu")
    batch_size: int = Field(default=16)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        if v < 1:
            raise ValueError("Batch size must be positive")
        return v

    @classmethod
    def from_environment(cls) -> "DeviceConfig":
        """Create config from environment settings"""
        device = os.getenv("DEVICE") or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        batch_size = int(
            os.getenv("BATCH_SIZE")
            or (64 if device == "mps" else 32 if device == "cuda" else 16)
        )
        return cls(device=device, batch_size=batch_size)


class ChromaConfig(BaseModel):
    """Configuration for ChromaDB"""

    persist_directory: str = Field(default="chroma_db")
    collection_name: str = Field(default="prediction_rag")
    anonymized_telemetry: bool = Field(default=False)
    is_persistent: bool = Field(default=True)
    hnsw_space: str = Field(default="cosine")
    construction_ef: int = Field(default=100)
    search_ef: int = Field(default=50)

    @field_validator("persist_directory")
    @classmethod
    def validate_persist_directory(cls, v):
        os.makedirs(v, exist_ok=True)
        return v


class ModelConfig(BaseModel):
    """Configuration for the model"""

    api_base: str = Field(default="https://openrouter.ai/api/v1")
    api_key: str
    model_name: str = Field(default="deepseek/deepseek-chat")
    max_tokens: int = Field(default=1500)
    temperature: float = Field(default=0.2)
    top_p: float = Field(default=0.9)
    frequency_penalty: float = Field(default=0.1)
    presence_penalty: float = Field(default=0.1)

    @field_validator("temperature", "top_p", "frequency_penalty", "presence_penalty")
    @classmethod
    def validate_float_range(cls, v, field):
        if not 0 <= v <= 1:
            raise ValueError(f"{field.name} must be between 0 and 1")
        return v


class AgentConfig(BaseModel):
    """Configuration for the agent"""

    max_iterations: int = Field(default=2)
    planning_interval: int = Field(default=1)
    verbose: bool = Field(default=True)


class SearchToolConfig(BaseModel):
    """Configuration for search tools"""

    max_results: int = Field(default=3)
    timeout: float = Field(default=10.0)


# Load environment variables
load_dotenv()

# Initialize configurations
try:
    device_config = DeviceConfig.from_environment()
    log_event(LogLevel.INFO, "Initialization", f"Using device: {device_config.device}")
except Exception as e:
    log_event(
        LogLevel.CRITICAL,
        "Initialization",
        "Failed to initialize device config",
        error=e,
    )
    raise

# Validate API keys
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    log_event(
        LogLevel.CRITICAL,
        "Configuration",
        "OPENROUTER_API_KEY not found in environment variables",
    )
    raise ValueError("OPENROUTER_API_KEY is required")

hf_token = os.getenv("HF_API_TOKEN")
if not hf_token:
    log_event(
        LogLevel.CRITICAL,
        "Configuration",
        "HF_API_TOKEN not found in environment variables",
    )
    raise ValueError("HF_API_TOKEN is required")

try:
    model_config = ModelConfig(api_key=api_key)
except Exception as e:
    log_event(
        LogLevel.CRITICAL, "Configuration", "Failed to initialize model config", error=e
    )
    raise


def create_prediction_agent() -> CodeAgent:
    """Create and configure the prediction agent"""

    # tools
    search_tool = DuckDuckGoSearchTool()

    # Initialize LiteLLM with OpenRouter
    model = LiteLLMModel(
        model_id="openrouter/deepseek/deepseek-chat",
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # Create agent with proper prompt
    agent = CodeAgent(
        model=model,
        tools=[search_tool],
        max_iterations=2,
        planning_interval=1,
        verbose=True,
        additional_authorized_imports=["json", "datetime"],
        system_prompt=CODE_SYSTEM_PROMPT + PREDICTION_AGENT_PROMPT,
        # add_base_tools=True,
    )

    return agent


def process_prediction_request(query: str) -> RAGState:
    """Process a prediction request"""

    # Initialize state
    state = RAGState(query=query)

    try:
        # Create agent
        agent = create_prediction_agent()

        # Format the task prompt with user's query
        task_prompt = AGENT_TASK_PROMPT.format(USER_PROMPT=query)
        response = agent.run(task_prompt)

        # Process response and update state
        try:
            console.print(Panel(response, title="Final Response", expand=False))

            # Parse structured response if needed
            # TODO: Add response parsing logic here if needed

        except Exception as e:
            log_event(
                LogLevel.ERROR, "Processing", "Failed to parse search results", error=e
            )
            state.add_error(f"Failed to process search results: {str(e)}")

        return state

    except Exception as e:
        log_event(LogLevel.CRITICAL, "Processing", "Failed to process request", error=e)
        state.add_error(str(e))
        return state
