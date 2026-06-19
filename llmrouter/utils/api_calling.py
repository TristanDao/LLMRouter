"""
API calling utilities using httpx for LLM API calls with manual round-robin load balancing.

This module provides functions for making API calls to LLM services (MiniMax, Alibaba, etc.)
using OpenAI-compatible endpoints with manual round-robin load balancing.
"""

import os
import json
import time
from typing import Dict, List, Union, Optional, Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

try:
    from transformers import GPT2TokenizerFast
except ImportError:
    GPT2TokenizerFast = None

# Global counter for round-robin API key selection
# Key: (api_endpoint, api_name) -> counter value
_api_key_counters: Dict[tuple, int] = {}
_gpt2_tokenizer = None


def _count_tokens(text: Optional[str]) -> int:
    """Fallback token counter using GPT-2 tokenizer if available."""
    if not text:
        return 0

    global _gpt2_tokenizer
    if GPT2TokenizerFast is None:
        return len(text.split())

    if _gpt2_tokenizer is None:
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    return len(_gpt2_tokenizer.encode(text))


def _parse_api_keys(api_keys_env: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Parse API keys from individual {SERVICE}_API_KEY environment variables.

    Supports: MINIMAX_API_KEY, ALIBABA_API_KEY, GEMINI_API_KEY, etc.
    The SERVICE part must match the 'service' field in the request (lowercase).

    Args:
        api_keys_env: Ignored (kept for backward compatibility)

    Returns:
        Dict[str, List[str]] mapping service name to list of API keys

    Raises:
        ValueError: If no API keys found
    """
    result: Dict[str, List[str]] = {}
    for env_key, env_val in os.environ.items():
        if env_key.endswith('_API_KEY') and env_val.strip():
            # Extract service name: MINIMAX_API_KEY -> minimax (lowercase)
            service = env_key[:-8].lower()
            result[service] = [env_val.strip()]

    if result:
        return result

    raise ValueError("No API keys found. Set {SERVICE}_API_KEY env vars (e.g., MINIMAX_API_KEY, ALIBABA_API_KEY)")


def _get_api_key(
    api_endpoint: str,
    api_name: str,
    api_keys: Union[Dict[str, List[str]], List[str]],
    service: Optional[str] = None,
    is_batch: bool = False,
    request_index: int = 0
) -> str:
    """
    Get an API key using round-robin selection.
    
    For dict format (service-based), uses service-specific keys and round-robin.
    For list format (legacy), uses all keys with round-robin.
    
    For single requests, uses a counter that increments per call.
    For batch requests, distributes requests across keys based on request_index.
    
    Args:
        api_endpoint: API endpoint URL (for counter key)
        api_name: API model name (for counter key)
        api_keys: Dict mapping service to keys, or List of keys (legacy format)
        service: Service provider name (e.g., "NVIDIA", "OpenAI"). Required if api_keys is dict.
        is_batch: Whether this is part of a batch request
        request_index: Index of the request in a batch (only used for batch requests)
    
    Returns:
        Selected API key string
    
    Raises:
        ValueError: If no keys available or service not found in dict
    """
    # Handle dict format (service-based)
    if isinstance(api_keys, dict):
        if not service:
            raise ValueError(
                "Service provider name is required when using dict format for API_KEYS. "
                "Please provide 'service' field in the request dict or ensure the model's "
                "llm_data includes a 'service' field matching a key in API_KEYS dict. "
                f"Available services in API_KEYS: {list(api_keys.keys())}"
            )
        
        # Normalize service name (case-insensitive)
        service_lower = service.lower()
        matching_service = None
        for key in api_keys.keys():
            if key.lower() == service_lower:
                matching_service = key
                break
        
        if matching_service is None:
            raise ValueError(
                f"Service '{service}' not found in API_KEYS dict. "
                f"Available services: {list(api_keys.keys())}. "
                "Please ensure the 'service' field in your request matches one of the keys in API_KEYS, "
                "or add the service to your API_KEYS configuration."
            )
        
        service_keys = api_keys[matching_service]
        if not service_keys:
            raise ValueError(f"No API keys available for service '{matching_service}' in API_KEYS dict")
        
        # Check if this is a local endpoint (localhost/127.0.0.1) with empty string key
        # Allow empty string for local providers
        is_local_endpoint = (
            "localhost" in api_endpoint.lower() or 
            "127.0.0.1" in api_endpoint or
            api_endpoint.startswith("http://127.0.0.1") or
            api_endpoint.startswith("http://localhost")
        )
        
        # If all keys are empty strings and it's a local endpoint, allow it
        if is_local_endpoint and all(key == "" for key in service_keys):
            return ""
        
        # For non-local endpoints or mixed keys, validate that we have non-empty keys
        if not any(key for key in service_keys):
            raise ValueError(
                f"No valid API keys available for service '{matching_service}' in API_KEYS dict. "
                f"Empty strings are only allowed for localhost endpoints."
            )
        
        # Filter out empty strings for non-local endpoints
        if not is_local_endpoint:
            service_keys = [key for key in service_keys if key]
            if not service_keys:
                raise ValueError(f"No valid API keys available for service '{matching_service}' in API_KEYS dict")
        
        # Use service-specific cache key for round-robin
        cache_key = (matching_service, api_endpoint, api_name)
        keys_to_use = service_keys
    
    # Handle list format (legacy)
    else:
        if not api_keys:
            raise ValueError("No API keys provided")
        
        # Check if this is a local endpoint with empty string key
        is_local_endpoint = (
            "localhost" in api_endpoint.lower() or 
            "127.0.0.1" in api_endpoint or
            api_endpoint.startswith("http://127.0.0.1") or
            api_endpoint.startswith("http://localhost")
        )
        
        # If all keys are empty strings and it's a local endpoint, allow it
        if is_local_endpoint and all(key == "" for key in api_keys):
            return ""
        
        # For non-local endpoints, filter out empty strings
        if not is_local_endpoint:
            api_keys = [key for key in api_keys if key]
            if not api_keys:
                raise ValueError("No valid API keys provided. Empty strings are only allowed for localhost endpoints.")
        
        cache_key = (api_endpoint, api_name)
        keys_to_use = api_keys
    
    # Perform round-robin selection
    if is_batch:
        # Batch request: distribute based on request_index
        selected_index = request_index % len(keys_to_use)
    else:
        # Single request: use counter and increment
        if cache_key not in _api_key_counters:
            _api_key_counters[cache_key] = 0
        
        selected_index = _api_key_counters[cache_key] % len(keys_to_use)
        _api_key_counters[cache_key] = (_api_key_counters[cache_key] + 1) % len(keys_to_use)
    
    return keys_to_use[selected_index]


def _make_request_with_retry(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int = 3
) -> httpx.Response:
    """
    Make HTTP request with exponential backoff retry.
    
    Args:
        url: Full API URL
        headers: Request headers
        payload: JSON payload
        timeout: Timeout in seconds
        max_retries: Maximum number of retries
    
    Returns:
        httpx.Response object
    
    Raises:
        httpx.HTTPStatusError: If all retries exhausted
    """
    import httpx
    
    for attempt in range(max_retries):
        try:
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            return response
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                time.sleep(wait_time)
                continue
            raise


def call_api(
    request: Union[Dict[str, Any], List[Dict[str, Any]]],
    api_keys_env: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.9,
    timeout: int = 60,
    max_retries: int = 3
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Call LLM API using httpx with OpenAI-compatible endpoints and manual round-robin load balancing.
    
    This function distributes API calls evenly across multiple API keys using
    manual round-robin selection. For batch requests, requests are distributed
    across API keys based on their index in the batch.
    
    Args:
        request: Single dict or list of dicts, each containing:
            - api_endpoint (str): API endpoint URL
            - query (str): The query/prompt to send
            - model_name (str): Model identifier name (not used for API call)
            - api_name (str): Actual API model name/path (e.g., "qwen/qwen2.5-7b-instruct")
            - service (str, optional): Service provider name (e.g., "NVIDIA", "OpenAI")
                                      Used for service-specific API key selection when API_KEYS is dict format
            - system_prompt (str, optional): System prompt for task-specific instructions
        api_keys_env: Optional override for API_KEYS env var (for testing)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        timeout: Request timeout in seconds
        max_retries: Maximum retries for failed requests
    
    Returns:
        Single dict or list of dicts (matching input format) with added fields:
            - response (str): API response text
            - token_num (int): Total tokens used
            - prompt_tokens (int): Input tokens
            - completion_tokens (int): Output tokens
            - response_time (float): Time taken in seconds
            - error (str, optional): Error message if request failed
    
    Example:
        Single request:
        >>> request = {
        ...     "api_endpoint": "https://integrate.api.nvidia.com/v1",
        ...     "query": "What is 2+2?",
        ...     "model_name": "qwen2.5-7b-instruct",
        ...     "api_name": "qwen/qwen2.5-7b-instruct"
        ... }
        >>> result = call_api(request)
        >>> print(result["response"])
        
        Batch requests (distributed across API keys):
        >>> requests = [request1, request2, request3]
        >>> results = call_api(requests)
    """
    if httpx is None:
        raise ImportError(
            "Missing required dependency `httpx` for API calls. "
            "Install it with: `pip install httpx`."
        )

    # Parse API keys from environment
    api_keys = _parse_api_keys(api_keys_env)
    
    # Handle single request vs batch
    is_single = isinstance(request, dict)
    requests = [request] if is_single else request
    
    # Validate request format
    required_keys = {'api_endpoint', 'query', 'model_name', 'api_name'}
    for req in requests:
        missing = required_keys - set(req.keys())
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
    
    results = []
    
    # Process each request
    for idx, req in enumerate(requests):
        result = req.copy()
        start_time = time.time()
        
        try:
            # Select API key using round-robin
            # For batch requests, use request index; for single requests, use counter
            # Extract service from request if available (for dict-based API key selection)
            service = req.get('service')
            selected_api_key = _get_api_key(
                api_endpoint=req['api_endpoint'],
                api_name=req['api_name'],
                api_keys=api_keys,
                service=service,
                is_batch=not is_single,
                request_index=idx
            )
            
            # Build messages list with optional system prompt
            messages = []
            if req.get('system_prompt'):
                messages.append({"role": "system", "content": req['system_prompt']})
            messages.append({"role": "user", "content": req['query']})
            
            # Build payload
            payload = {
                "model": req['api_name'],
                "messages": messages,
                "max_tokens": req.get('max_tokens', max_tokens),
                "temperature": temperature,
            }
            
            # Disable thinking for MiniMax models (they output thinking tags by default)
            service_lower = req.get('service', '').lower()
            if service_lower == 'minimax':
                payload["thinking"] = {"type": "disabled"}
            
            # Add optional parameters
            if top_p and top_p != 0.9:
                payload["top_p"] = top_p
            
            # Build headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {selected_api_key}"
            }
            
            # Build full URL
            url = req['api_endpoint']
            if not url.endswith('/'):
                url += '/'
            url += 'chat/completions'
            
            # Make request with retry
            response = _make_request_with_retry(
                url=url,
                headers=headers,
                payload=payload,
                timeout=timeout,
                max_retries=max_retries
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
                result['response'] = f"API Error: {error_msg}"
                result['token_num'] = 0
                result['prompt_tokens'] = 0
                result['completion_tokens'] = 0
                result['response_time'] = time.time() - start_time
                result['error'] = error_msg
                results.append(result)
                continue
            
            # Parse response
            resp_data = response.json()
            
            # Extract response text
            choices = resp_data.get("choices", [])
            if choices:
                response_text = choices[0].get("message", {}).get("content", "")
            else:
                response_text = ""
            
            # Extract token counts from usage
            usage = resp_data.get("usage", {})
            token_num = usage.get("total_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            # Fallback estimation if usage not provided
            if token_num == 0:
                prompt_tokens = _count_tokens(req.get('query'))
                completion_tokens = _count_tokens(response_text)
                token_num = prompt_tokens + completion_tokens
            
            end_time = time.time()
            
            # Add results to response
            result['response'] = response_text
            result['token_num'] = token_num
            result['prompt_tokens'] = prompt_tokens
            result['completion_tokens'] = completion_tokens
            result['response_time'] = end_time - start_time
            
        except Exception as e:
            error_msg = str(e)
            end_time = time.time()
            
            # Add error information
            result['response'] = f"API Error: {error_msg[:200]}"
            result['token_num'] = 0
            result['prompt_tokens'] = 0
            result['completion_tokens'] = 0
            result['response_time'] = end_time - start_time
            result['error'] = error_msg
        
        results.append(result)
    
    # Return single result or list based on input
    return results[0] if is_single else results
