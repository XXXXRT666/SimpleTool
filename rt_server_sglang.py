#!/usr/bin/env python3
"""
SimpleTool SGLang Server - Multi-Head Parallel Decoding for Real-Time Function Calling
"""

import asyncio
import inspect
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_download import DEFAULT_MODEL_PATH, ensure_default_model

try:
    from sglang import Engine as SGLangEngine
except ImportError:
    # Backward-compat path for older SGLang package layouts.
    from sglang.srt.entrypoints.engine import Engine as SGLangEngine

# ==================== Config ====================
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8899"))
MAX_HISTORY = 6

ENABLE_TORCH_COMPILE = os.environ.get("SGLANG_ENABLE_TORCH_COMPILE", "1").lower() not in {
    "0",
    "false",
    "no",
}
ENABLE_RADIX_CACHE = os.environ.get("SGLANG_ENABLE_RADIX_CACHE", "1").lower() not in {
    "0",
    "false",
    "no",
}

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ==================== Multi-Head Tags ====================
HEAD_TAGS = ["<content>", "<function>", "<arg1>", "<arg2>", "<arg3>", "<arg4>", "<arg5>", "<arg6>"]
STOP_TOKENS = [
    "<|null|>",
    "</content>",
    "</function>",
    "</arg1>",
    "</arg2>",
    "</arg3>",
    "</arg4>",
    "</arg5>",
    "</arg6>",
    "<|im_end|>",
]

SYSTEM_TEMPLATE = """<|im_start|>system
You are a multi-head parallel function calling model.
## Output Heads

**Head 0 - <content>**: Natural language response
- Format: <content>response text</content>

**Head 1 - <function>**: Function names to call
- Format: <function>name</function>

**Head 2-7 - <arg1>-<arg6>**: Function arguments by position
- Format: <argN>value</argN>
- If Unnecessary: <argN><|null|></argN>

## Available Tools:

{tools_json}
<|im_end|>
"""


# ==================== Data Models ====================
class Message(BaseModel):
    role: str
    content: str


class FCRequest(BaseModel):
    messages: List[Message]
    tools: List[Dict[str, Any]]
    environment: Optional[List[str]] = None
    history: Optional[List[str]] = None
    max_tokens: int = 32
    temperature: float = 0.0
    include_content_head: bool = False


class FCResponse(BaseModel):
    success: bool
    function: Optional[str] = None
    args: Dict[str, Any] = {}
    heads: Dict[str, str] = {}
    content: Optional[str] = None
    latency_ms: float = 0
    error: Optional[str] = None


# ==================== SimpleTool Engine ====================
class SimpleToolEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.engine: Optional[SGLangEngine] = None

    def _build_engine_kwargs(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "trust_remote_code": True,
            "tp_size": 1,
            "mem_fraction_static": 0.8,
            "context_length": 1024,
            "enable_torch_compile": True,
            "quantization": "w8a8_int8",
        }

    def initialize(self):
        self.model_path = ensure_default_model(self.model_path)
        print(f"[SimpleTool-SGLang] Loading model: {self.model_path}")
        print(f"[SimpleTool-SGLang] Flags: torch_compile={ENABLE_TORCH_COMPILE}, radix_cache={ENABLE_RADIX_CACHE}")
        engine_kwargs = self._build_engine_kwargs()
        self.engine = SGLangEngine(**engine_kwargs)
        print("[SimpleTool-SGLang] Model loaded!")
        print("[SimpleTool-SGLang] Warmup skipped.")

    def _warmup(self):
        print("[SimpleTool-SGLang] Warming up...")
        dummy_tools = '[{"type":"function","function":{"name":"test","parameters":{}}}]'
        prompt = SYSTEM_TEMPLATE.format(tools_json=dummy_tools)
        prompt += "<|im_start|>user\nenvironment: []\nhistory: []\n\ntest<|im_end|>\n<|im_start|>assistant\n"
        prompts = [prompt + tag for tag in HEAD_TAGS]
        self._generate_batch(prompts, max_tokens=32, temperature=0.0)
        print("[SimpleTool-SGLang] Warmup complete!")

    def _build_tools_json(self, tools: List[Dict]) -> str:
        return "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)

    def _extract_param_info(self, tools: List[Dict]) -> List[str]:
        names = []
        for tool in tools:
            func = tool.get("function", {})
            params = func.get("parameters", {}).get("properties", {})
            for name in params.keys():
                if name not in names:
                    names.append(name)
        return names[:6]

    def _get_max_args(self, tools: List[Dict]) -> int:
        max_args = 0
        for tool in tools:
            func = tool.get("function", {})
            params = func.get("parameters", {}).get("properties", {})
            max_args = max(max_args, len(params))
        return min(max_args, 6)

    def _extract_text(self, output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            for key in ("text", "output_text", "generated_text", "completion"):
                value = output.get(key)
                if isinstance(value, str):
                    return value
            choices = output.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        return message["content"]
                    if isinstance(first.get("text"), str):
                        return first["text"]

        for attr in ("text", "output_text", "generated_text"):
            if hasattr(output, attr):
                value = getattr(output, attr)
                if isinstance(value, str):
                    return value

        if hasattr(output, "outputs"):
            outputs = getattr(output, "outputs")
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if hasattr(first, "text"):
                    value = getattr(first, "text")
                    if isinstance(value, str):
                        return value

        return str(output)

    def _normalize_batch_output(self, raw: Any, expected_count: int) -> List[str]:
        if isinstance(raw, list):
            if len(raw) == expected_count:
                return [self._extract_text(item) for item in raw]
            if len(raw) == 1:
                one = raw[0]
                if isinstance(one, list) and len(one) == expected_count:
                    return [self._extract_text(item) for item in one]

        if isinstance(raw, dict):
            texts = raw.get("texts")
            if isinstance(texts, list) and len(texts) == expected_count:
                return [self._extract_text(item) for item in texts]
            if isinstance(raw.get("text"), list) and len(raw["text"]) == expected_count:
                return [self._extract_text(item) for item in raw["text"]]

        raise RuntimeError(f"Unexpected SGLang batch output type: {type(raw)!r}. Expected {expected_count} results.")

    def _invoke_generate(self, *args, **kwargs) -> Any:
        assert self.engine is not None
        result = self.engine.generate(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return result

    def _generate_single(self, prompt: str, sampling_params: Dict[str, Any]) -> str:
        attempts = [
            lambda: self._invoke_generate(prompt, sampling_params=sampling_params),
            lambda: self._invoke_generate(prompt, sampling_params),
        ]

        errors: List[str] = []
        for attempt in attempts:
            try:
                result = attempt()
                return self._extract_text(result)
            except Exception as exc:
                errors.append(str(exc))

        raise RuntimeError("; ".join(errors))

    def _generate_batch(self, prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
        sampling_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop": STOP_TOKENS,
            "skip_special_tokens": False,
        }

        attempts = [
            lambda: self._invoke_generate(prompts, sampling_params=sampling_params),
            lambda: self._invoke_generate(prompts, sampling_params),
        ]

        errors: List[str] = []
        for attempt in attempts:
            try:
                raw = attempt()
                return self._normalize_batch_output(raw, expected_count=len(prompts))
            except Exception as exc:
                errors.append(str(exc))

        # Fallback for versions that only support single prompt calls.
        texts = []
        for prompt in prompts:
            texts.append(self._generate_single(prompt, sampling_params))
        if len(texts) == len(prompts):
            return texts

        raise RuntimeError("; ".join(errors))

    async def _invoke_generate_async(self, *args, **kwargs) -> Any:
        assert self.engine is not None

        async_generate = getattr(self.engine, "async_generate", None)
        if async_generate is not None:
            result = async_generate(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result

        generate = getattr(self.engine, "generate")
        if inspect.iscoroutinefunction(generate):
            return await generate(*args, **kwargs)

        result = await asyncio.to_thread(generate, *args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _generate_single_async(self, prompt: str, sampling_params: Dict[str, Any]) -> str:
        attempts = [
            lambda: self._invoke_generate_async(prompt, sampling_params=sampling_params),
            lambda: self._invoke_generate_async(prompt, sampling_params),
        ]

        errors: List[str] = []
        for attempt in attempts:
            try:
                result = await attempt()
                return self._extract_text(result)
            except Exception as exc:
                errors.append(str(exc))

        raise RuntimeError("; ".join(errors))

    async def _generate_batch_async(self, prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
        sampling_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop": STOP_TOKENS,
            "skip_special_tokens": False,
        }

        attempts = [
            lambda: self._invoke_generate_async(prompts, sampling_params=sampling_params),
            lambda: self._invoke_generate_async(prompts, sampling_params),
        ]

        errors: List[str] = []
        for attempt in attempts:
            try:
                raw = await attempt()
                return self._normalize_batch_output(raw, expected_count=len(prompts))
            except Exception as exc:
                errors.append(str(exc))

        texts = []
        for prompt in prompts:
            texts.append(await self._generate_single_async(prompt, sampling_params))
        if len(texts) == len(prompts):
            return texts

        raise RuntimeError("; ".join(errors))

    def _build_generation_inputs(self, request: FCRequest) -> Tuple[List[str], int]:
        tools_json = self._build_tools_json(request.tools)
        system_prompt = SYSTEM_TEMPLATE.format(tools_json=tools_json)

        env_str = json.dumps(request.environment or [], ensure_ascii=False)
        hist_list = (request.history or [])[-MAX_HISTORY:]
        hist_str = ", ".join(hist_list) if hist_list else ""

        query = ""
        for msg in request.messages:
            if msg.role == "user":
                query = msg.content

        user_turn = (
            f"<|im_start|>user\nenvironment: {env_str}\nhistory: [{hist_str}]\n\n"
            f"{query}<|im_end|>\n<|im_start|>assistant\n"
        )
        full_prefix = system_prompt + user_turn

        max_args = self._get_max_args(request.tools)
        active_tags = ["<function>"] + [f"<arg{i}>" for i in range(1, max_args + 1)]
        if request.include_content_head:
            active_tags = ["<content>"] + active_tags

        prompts = [full_prefix + tag for tag in active_tags]
        return prompts, max_args

    def _build_response(self, request: FCRequest, texts: List[str], max_args: int, start: float) -> FCResponse:
        latency_ms = (time.perf_counter() - start) * 1000

        heads: Dict[str, str] = {}
        head_names = []
        if request.include_content_head:
            head_names.append("content")
        head_names.append("function")
        head_names.extend([f"arg{i}" for i in range(1, max_args + 1)])

        for i, text in enumerate(texts):
            cleaned = text.strip()
            for stop in STOP_TOKENS:
                if cleaned.endswith(stop):
                    cleaned = cleaned[: -len(stop)].strip()
                    break
            heads[head_names[i]] = cleaned

        func_name = heads.get("function", "").strip()
        if not func_name or func_name == "<|null|>":
            return FCResponse(
                success=False,
                heads=heads,
                content=heads.get("content"),
                latency_ms=latency_ms,
                error="No function called",
            )

        param_names = self._extract_param_info(request.tools)
        args: Dict[str, Any] = {}
        for i, name in enumerate(param_names):
            val = heads.get(f"arg{i + 1}", "").strip()
            if val and val != "<|null|>":
                if val.isdigit():
                    args[name] = int(val)
                elif val.lstrip("-").replace(".", "", 1).isdigit():
                    args[name] = float(val)
                else:
                    args[name] = val.lower().strip()

        return FCResponse(
            success=True,
            function=func_name,
            args=args,
            heads=heads,
            content=heads.get("content"),
            latency_ms=latency_ms,
        )

    def call(self, request: FCRequest) -> FCResponse:
        start = time.perf_counter()
        prompts, max_args = self._build_generation_inputs(request)
        texts = self._generate_batch(
            prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return self._build_response(request, texts, max_args, start)

    async def call_async(self, request: FCRequest) -> FCResponse:
        start = time.perf_counter()
        prompts, max_args = self._build_generation_inputs(request)
        texts = await self._generate_batch_async(
            prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return self._build_response(request, texts, max_args, start)


# ==================== FastAPI ====================
engine: Optional[SimpleToolEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SimpleToolEngine(MODEL_PATH)
    engine.initialize()
    yield
    print("[Server] Shutdown")


app = FastAPI(title="SimpleTool SGLang Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded": engine is not None and engine.engine is not None,
        "backend": "sglang",
        "model": MODEL_PATH,
        "torch_compile": ENABLE_TORCH_COMPILE,
        "radix_cache": ENABLE_RADIX_CACHE,
    }


@app.post("/v1/function_call", response_model=FCResponse)
async def function_call(request: FCRequest):
    if engine is None or engine.engine is None:
        raise HTTPException(503, "Model not loaded")
    try:
        return await engine.call_async(request)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return FCResponse(success=False, error=str(e), latency_ms=0)


if __name__ == "__main__":
    print(
        r"""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   ███████╗██╗███╗   ███╗██████╗ ██╗     ███████╗                   ║
║   ██╔════╝██║████╗ ████║██╔══██╗██║     ██╔════╝                   ║
║   ███████╗██║██╔████╔██║██████╔╝██║     █████╗                     ║
║   ╚════██║██║██║╚██╔╝██║██╔═══╝ ██║     ██╔══╝                     ║
║   ███████║██║██║ ╚═╝ ██║██║     ███████╗███████╗                   ║
║   ╚══════╝╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝                   ║
║                                                                    ║
║          SimpleTool SGLang-Server v1.0                             ║
║          Having a Realtime LLM based control time!                 ║
║                                                                    ║
║   Default model: ./models/RT-Qwen3-4B-AWQ                          ║
║   Default flags: torch_compile=on, radix_cache=on                  ║
║   Run Demos: Open demos/*.html in browser                          ║
║   Build New: Send simpletool-game-guide.md to AI(Claude Gemini...) ║
║              for Building new your own HTML games easily           ║
║   Endpoints:                                                       ║
║     GET  /health           - Health check                          ║
║     POST /v1/function_call - Function call API                     ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """
    )
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
