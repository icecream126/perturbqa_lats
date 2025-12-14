import os
import warnings
import json
import torch

# Ensure unsloth (if present) is imported before transformers for optimizations
try:
    from unsloth import FastLanguageModel  # noqa: F401
    _HAS_UNSLOTH = True
except ImportError:
    _HAS_UNSLOTH = False

import backoff
from openai import OpenAI, OpenAIError
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline

completion_tokens = prompt_tokens = 0
MAX_TOKENS = 4000
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

# Cache local pipelines to avoid reloading models repeatedly
_local_pipelines = {}
_local_tokenizers = {}
_local_kinds = {}  # "hf_pipe" or "unsloth"

def tokens_in_text(text):
    """
    Accurately count the number of tokens in a string using the GPT-2 tokenizer.
    
    :param text: The input text.
    :return: The exact number of tokens in the text.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.encode(text)
    return len(tokens)

_client = None


def _get_openai_client():
    """
    Lazily build an OpenAI client only when an OpenAI backend is used.
    If no API key is present, instruct the user to switch to --backend local.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")

    if api_key == "":
        raise OpenAIError(
            "OPENAI_API_KEY is required for OpenAI backends. "
            "Use --backend local (and set LOCAL_MODEL_NAME) to avoid OpenAI."
        )

    client_kwargs = {"api_key": api_key}
    if api_base != "":
        print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
        client_kwargs["base_url"] = api_base

    _client = OpenAI(**client_kwargs)
    return _client

@backoff.on_exception(backoff.expo, OpenAIError)
def completions_with_backoff(**kwargs):
    return _get_openai_client().chat.completions.create(**kwargs)

def _parse_bool_env(env_name: str, default: bool = False) -> bool:
    val = os.getenv(env_name, None)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "y", "on")


def _get_local_pipeline(model_name: str):
    """Load and cache a local generation backend."""
    if model_name in _local_pipelines:
        return _local_pipelines[model_name], _local_tokenizers[model_name], _local_kinds[model_name]

    is_unsloth_model = "unsloth" in model_name.lower()

    # --- Unsloth path ---
    if is_unsloth_model and _HAS_UNSLOTH:
        print("Loading Unsloth Model...")
        print(f"   Detected Unsloth model: {model_name}")

        actual_model_name = model_name
        is_likely_pre_quantized = any(indicator in model_name.lower() for indicator in [
            "kimi", "thinking", "compressed", "ct"
        ])

        # Gather optional settings from env
        max_seq_length = int(os.getenv("LOCAL_MAX_SEQ_LENGTH", "32768"))
        dtype_env = os.getenv("LOCAL_DTYPE", None)
        load_in_4bit_env = _parse_bool_env("LOCAL_LOAD_IN_4BIT", False)

        model_kwargs = {
            "model_name": actual_model_name,
            "max_seq_length": max_seq_length,
        }
        if dtype_env:
            try:
                model_kwargs["dtype"] = getattr(torch, dtype_env)
            except AttributeError:
                print(f"⚠️  Unknown dtype '{dtype_env}', ignoring.")
        if is_likely_pre_quantized and not load_in_4bit_env:
            pass
        elif load_in_4bit_env:
            model_kwargs["load_in_4bit"] = True
            print("⚠️  Using 4-bit quantization (may conflict with pre-quantized models)")
        else:
            model_kwargs["load_in_4bit"] = False

        try:
            print(f"   Loading model with kwargs: {json.dumps(list(model_kwargs.keys()))}")
            print(f"   Model name: {actual_model_name}")
            print(f"   Max seq length: {model_kwargs['max_seq_length']}")
            print(f"   Load in 4bit: {model_kwargs.get('load_in_4bit', 'not set')}")

            model, tok = FastLanguageModel.from_pretrained(**model_kwargs)
            if model is None or tok is None:
                raise ValueError(f"FastLanguageModel.from_pretrained returned None. Model: {model}, Tokenizer: {tok}")

            FastLanguageModel.for_inference(model)

            print("   Warming up model for thread-safe batch processing...")
            try:
                with torch.inference_mode():
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    dummy_input = tok("test", return_tensors="pt").to(device)
                    model.to(device)
                    _ = model.generate(**dummy_input, max_new_tokens=1, do_sample=False)
                print("   ✓ Model warm-up completed")
            except Exception as warmup_error:
                print(f"   ⚠️  Model warm-up failed (non-critical): {warmup_error}")

            _local_pipelines[model_name] = model
            _local_tokenizers[model_name] = tok
            _local_kinds[model_name] = "unsloth"
            return model, tok, "unsloth"
        except Exception as e:
            if "CompressedTensorsConfig" in str(e) or "BitsAndBytesConfig" in str(e):
                print("⚠️  Detected quantization conflict. Trying alternative loading method...")
                try:
                    clean_kwargs = {
                        "model_name": actual_model_name,
                        "max_seq_length": max_seq_length,
                    }
                    if dtype_env:
                        try:
                            clean_kwargs["dtype"] = getattr(torch, dtype_env)
                        except AttributeError:
                            pass
                    print("   Attempting to load without quantization parameters...")
                    model, tok = FastLanguageModel.from_pretrained(**clean_kwargs)
                    FastLanguageModel.for_inference(model)
                    print("   Warming up model for thread-safe batch processing...")
                    try:
                        with torch.inference_mode():
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            dummy_input = tok("test", return_tensors="pt").to(device)
                            model.to(device)
                            _ = model.generate(**dummy_input, max_new_tokens=1, do_sample=False)
                        print("   ✓ Model warm-up completed")
                    except Exception as warmup_error:
                        print(f"   ⚠️  Model warm-up failed (non-critical): {warmup_error}")

                    _local_pipelines[model_name] = model
                    _local_tokenizers[model_name] = tok
                    _local_kinds[model_name] = "unsloth"
                    return model, tok, "unsloth"
                except Exception as e2:
                    print(f"❌ Failed to load model: {e2}")
                    raise ValueError(f"Could not load model. Original error: {e}. Retry error: {e2}")
            else:
                print(f"❌ Failed to load Unsloth model: {e}")
                raise

    # --- Default HF path ---
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    pipe = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        device_map="auto",
    )
    _local_pipelines[model_name] = pipe
    _local_tokenizers[model_name] = tok
    _local_kinds[model_name] = "hf_pipe"
    return pipe, tok, "hf_pipe"

def _chat_local(prompt, model_name, temperature=1.0, max_tokens=100, n=1, stop=None):
    backend, tok, kind = _get_local_pipeline(model_name)
    results = []

    if kind == "unsloth":
        model = backend
        device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = tok(prompt, return_tensors="pt").to(device)
        gen_outputs = model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            num_return_sequences=n,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        for i in range(n):
            seq = gen_outputs[i]
            gen_text = tok.decode(seq[input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
            if stop:
                stops = stop if isinstance(stop, list) else [stop]
                min_idx = len(gen_text)
                for s in stops:
                    idx = gen_text.find(s)
                    if idx != -1:
                        min_idx = min(min_idx, idx)
                gen_text = gen_text[:min_idx]
            results.append(gen_text)
        return results

    # HF pipeline path
    outputs = backend(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        num_return_sequences=n,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    for out in outputs:
        text = out["generated_text"][len(prompt):]
        if stop:
            stops = stop if isinstance(stop, list) else [stop]
            min_idx = len(text)
            for s in stops:
                idx = text.find(s)
                if idx != -1:
                    min_idx = min(min_idx, idx)
            text = text[:min_idx]
        results.append(text)
    return results

def gpt(prompt, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, n=1, stop=None, local_model_name=None) -> list:
    """
    Unified GPT interface.
    - For model == 'local', run HF pipeline (local_model_name or env LOCAL_MODEL_NAME).
    - Otherwise, call OpenAI ChatCompletion.
    """
    if model == "local":
        target_model = local_model_name or os.getenv("LOCAL_MODEL_NAME", "unsloth/DeepSeek-R1-Distill-Llama-8B")
        return _chat_local(prompt, target_model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    # DEBUG: LLM API 호출 직전 확인
    # 확인할 값: prompt (최종 프롬프트 문자열), model, messages 구조
    import pdb; pdb.set_trace()
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-3.5-turbo-16k":
        cost = completion_tokens / 1000 * 0.004 + prompt_tokens / 1000 * 0.003
    else:
        cost = 0
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
