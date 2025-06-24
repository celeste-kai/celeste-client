"""
Optimized Gemini API Client for Alita.

This module provides a production-ready, high-performance client for interacting 
with the Gemini 2.5 API following 2024 best practices for security, performance,
reliability, and observability.
"""

import os
import json
import time
import uuid
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, AsyncContextManager
from contextlib import asynccontextmanager
import re

import aiohttp
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log
)


# Custom Exception Hierarchy
class GeminiClientError(Exception):
    """Base exception for Gemini client errors."""
    pass


class GeminiAuthenticationError(GeminiClientError):
    """Authentication/authorization errors."""
    pass


class GeminiRateLimitError(GeminiClientError):
    """Rate limiting errors."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class GeminiValidationError(GeminiClientError):
    """Input validation errors."""
    pass


class GeminiAPIError(GeminiClientError):
    """API-specific errors."""
    
    def __init__(self, message: str, status_code: int, error_details: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_details = error_details or {}


class GeminiNetworkError(GeminiClientError):
    """Network connectivity errors."""
    pass


@dataclass
class ClientConfig:
    """Configuration for Gemini client with validation."""
    
    api_key: Optional[str] = None
    #model: str = "gemini-2.5-flash-preview-05-20"
    model: str = "gemini-2.0-flash"
    temperature: float = 0.2
    max_output_tokens: int = 8192
    top_p: float = 0.95
    top_k: int = 40
    timeout: float = 30.0
    base_url: str = "https://generativelanguage.googleapis.com"
    api_version: str = "v1beta"
    max_retries: int = 3
    connection_pool_size: int = 100
    connection_pool_maxsize: int = 100
    keepalive_timeout: int = 30
    enable_compression: bool = True
    log_level: str = "INFO"
    mask_credentials: bool = True
    correlation_id_header: str = "X-Correlation-ID"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_config()
        
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("GOOGLE_API_KEY")
            
        if not self.api_key:
            raise GeminiValidationError(
                "API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )
            
        # Validate API key format
        if not self._is_valid_api_key(self.api_key):
            raise GeminiValidationError("Invalid API key format.")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.temperature <= 1.0:
            raise GeminiValidationError("Temperature must be between 0.0 and 1.0")
            
        if not 0.0 <= self.top_p <= 1.0:
            raise GeminiValidationError("top_p must be between 0.0 and 1.0")
            
        if self.top_k < 1:
            raise GeminiValidationError("top_k must be >= 1")
            
        if self.max_output_tokens < 1:
            raise GeminiValidationError("max_output_tokens must be >= 1")
            
        if self.timeout <= 0:
            raise GeminiValidationError("timeout must be > 0")
            
        if self.max_retries < 0:
            raise GeminiValidationError("max_retries must be >= 0")
    
    def _is_valid_api_key(self, api_key: str) -> bool:
        """Validate API key format."""
        # Basic validation - Gemini API keys typically start with specific patterns
        return bool(api_key and len(api_key) > 10 and api_key.replace("-", "").replace("_", "").isalnum())
    
    def get_masked_api_key(self) -> str:
        """Return masked API key for logging."""
        if not self.api_key or len(self.api_key) < 8:
            return "***"
        return f"{self.api_key[:4]}***{self.api_key[-4:]}"


@dataclass
class RequestMetrics:
    """Metrics for a request."""
    
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None
    retry_count: int = 0
    error: Optional[str] = None
    
    def complete(self, status_code: int, error: Optional[str] = None):
        """Mark request as completed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status_code = status_code
        self.error = error


class GeminiClient:
    """
    Production-ready Gemini API client with comprehensive error handling,
    performance optimization, and security best practices.
    """
    
    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """Initialize the optimized Gemini client.
        
        Args:
            config: Client configuration object
            logger: Logger instance
            **kwargs: Additional configuration parameters
        """
        # Merge kwargs into config
        config_dict = {}
        if config:
            config_dict.update(config.__dict__)
        config_dict.update(kwargs)
        
        self.config = ClientConfig(**config_dict)
        
        # Set up logging
        self.logger = logger or self._setup_logger()
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        # Metrics tracking
        self._request_metrics: List[RequestMetrics] = []
        self._total_requests = 0
        self._total_errors = 0
        
        self.logger.info(
            f"Gemini client initialized with model={self.config.model}, "
            f"api_key={self.config.get_masked_api_key()}"
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Set up structured logging."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper connection pooling."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                # Double-check pattern for race condition safety
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=self.config.connection_pool_size,
                        limit_per_host=self.config.connection_pool_maxsize,
                        keepalive_timeout=self.config.keepalive_timeout,
                        enable_cleanup_closed=True
                    )
                    
                    timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                    
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={"User-Agent": "Alita-GeminiClient/1.0"}
                    )
                    
                    self.logger.debug("Created new aiohttp session with optimized configuration")
        
        return self._session
    
    async def close(self):
        """Close the client session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            self.logger.debug("Gemini client session closed")
    
    async def __aenter__(self) -> "GeminiClient":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()
    
    def _build_url(self, endpoint: str) -> str:
        """Build API URL without exposing API key in logs."""
        return f"{self.config.base_url}/{self.config.api_version}/{endpoint}"
    
    def _build_headers(self, correlation_id: Optional[str] = None) -> Dict[str, str]:
        """Build request headers with security considerations."""
        headers = {
            "Content-Type": "application/json",
        }
        
        if correlation_id:
            headers[self.config.correlation_id_header] = correlation_id
            
        return headers
    
    def _build_request_payload(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> Dict[str, Any]:
        """Build the API request payload."""
        generation_config = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "topP": kwargs.get("top_p", self.config.top_p),
            "topK": kwargs.get("top_k", self.config.top_k),
            "maxOutputTokens": kwargs.get("max_output_tokens", self.config.max_output_tokens),
        }
        
        return {
            "contents": messages,
            "generationConfig": generation_config,
        }
    
    def _sanitize_for_logging(self, data: Any) -> Any:
        """Sanitize sensitive data for logging."""
        if not self.config.mask_credentials:
            return data
            
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in ['key', 'token', 'auth', 'password', 'secret']):
                    sanitized[key] = "***MASKED***"
                else:
                    sanitized[key] = self._sanitize_for_logging(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_for_logging(item) for item in data]
        elif isinstance(data, str) and len(data) > 100:
            return f"{data[:50]}...{data[-50:]}"
        
        return data
    
    def _classify_error(self, status_code: int, response_data: Dict) -> GeminiClientError:
        """Classify HTTP errors into appropriate exception types."""
        if status_code == 401:
            return GeminiAuthenticationError("Invalid API key or authentication failed")
        elif status_code == 403:
            return GeminiAuthenticationError("Access forbidden - check API permissions")
        elif status_code == 429:
            retry_after = response_data.get("retry_after", 60)
            return GeminiRateLimitError(
                "Rate limit exceeded",
                retry_after=retry_after
            )
        elif 400 <= status_code < 500:
            return GeminiValidationError(f"Client error: {response_data}")
        elif 500 <= status_code < 600:
            return GeminiAPIError(
                f"Server error: {status_code}",
                status_code=status_code,
                error_details=response_data
            )
        else:
            return GeminiAPIError(
                f"Unexpected status code: {status_code}",
                status_code=status_code,
                error_details=response_data
            )
    
    def _is_retryable_error(self, exception: Exception) -> bool:
        """Determine if an error should be retried."""
        # Retry on network errors and certain API errors
        if isinstance(exception, (aiohttp.ClientError, asyncio.TimeoutError)):
            return True
        
        if isinstance(exception, GeminiRateLimitError):
            return True
            
        if isinstance(exception, GeminiAPIError) and exception.status_code >= 500:
            return True
            
        return False
    
    # Note: Retry logic is handled manually within the method to support dynamic config
    async def generate_content(
        self, 
        messages: List[Dict[str, Any]], 
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using the Gemini API with comprehensive error handling.
        
        Args:
            messages: List of message dictionaries with roles and content
            correlation_id: Optional correlation ID for request tracing
            **kwargs: Additional parameters to override defaults
            
        Returns:
            API response as a dictionary
            
        Raises:
            GeminiValidationError: If input validation fails
            GeminiAuthenticationError: If authentication fails
            GeminiRateLimitError: If rate limited
            GeminiAPIError: For API-specific errors
            GeminiNetworkError: For network connectivity issues
        """
        # Input validation
        if not messages:
            raise GeminiValidationError("Messages cannot be empty")
        
        for msg in messages:
            if "role" not in msg or "parts" not in msg:
                raise GeminiValidationError("Each message must contain 'role' and 'parts' keys")
        
        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Create metrics tracker
        metrics = RequestMetrics(
            request_id=correlation_id,
            start_time=time.time()
        )
        
        self._total_requests += 1
        
        # Manual retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                session = await self._get_session()
                model = kwargs.get("model", self.config.model)
                url = self._build_url(f"models/{model}:generateContent")
                url += f"?key={self.config.api_key}"  # Gemini uses query parameter for API key
                headers = self._build_headers(correlation_id)
                payload = self._build_request_payload(messages, **kwargs)
                
                # Log request (sanitized)
                self.logger.debug(
                    f"Sending request [ID: {correlation_id}] to {model} (attempt {attempt + 1})",
                    extra={
                        "correlation_id": correlation_id,
                        "model": model,
                        "payload_size": len(json.dumps(payload)),
                        "message_count": len(messages),
                        "attempt": attempt + 1
                    }
                )
                
                async with session.post(
                    url,
                    json=payload,
                    headers=headers
                ) as response:
                    response_data = await response.json()
                    
                    if response.status != 200:
                        error = self._classify_error(response.status, response_data)
                        
                        # Check if this is the last attempt or non-retryable error
                        if attempt == self.config.max_retries or not self._is_retryable_error(error):
                            metrics.complete(response.status, str(error))
                            self._request_metrics.append(metrics)
                            self._total_errors += 1
                            
                            self.logger.error(
                                f"API error [ID: {correlation_id}]: {response.status} (final attempt)",
                                extra={
                                    "correlation_id": correlation_id,
                                    "status_code": response.status,
                                    "error_type": type(error).__name__,
                                    "attempt": attempt + 1
                                }
                            )
                            raise error
                        else:
                            # Log retry attempt
                            self.logger.warning(
                                f"Retryable error [ID: {correlation_id}]: {response.status}, retrying...",
                                extra={
                                    "correlation_id": correlation_id,
                                    "status_code": response.status,
                                    "error_type": type(error).__name__,
                                    "attempt": attempt + 1,
                                    "retries_left": self.config.max_retries - attempt
                                }
                            )
                            
                            # Store for potential re-raise
                            last_exception = error
                            
                            # Exponential backoff with jitter
                            delay = min(2 ** attempt + (attempt * 0.1), 30)
                            await asyncio.sleep(delay)
                            continue
                    
                    # Success
                    metrics.complete(response.status)
                    self._request_metrics.append(metrics)
                    
                    self.logger.debug(
                        f"Request completed successfully [ID: {correlation_id}] (attempt {attempt + 1})",
                        extra={
                            "correlation_id": correlation_id,
                            "duration": metrics.duration,
                            "status_code": response.status,
                            "attempt": attempt + 1
                        }
                    )
                    
                    return response_data
                    
            except aiohttp.ClientError as e:
                last_exception = GeminiNetworkError(f"Network error: {e}")
                
                # Check if this is the last attempt
                if attempt == self.config.max_retries:
                    metrics.complete(0, str(e))
                    self._request_metrics.append(metrics)
                    self._total_errors += 1
                    
                    self.logger.error(
                        f"Network error [ID: {correlation_id}]: {e} (final attempt)",
                        extra={"correlation_id": correlation_id, "attempt": attempt + 1}
                    )
                    raise last_exception from e
                else:
                    # Log retry attempt
                    self.logger.warning(
                        f"Network error [ID: {correlation_id}]: {e}, retrying...",
                        extra={
                            "correlation_id": correlation_id,
                            "attempt": attempt + 1,
                            "retries_left": self.config.max_retries - attempt
                        }
                    )
                    
                    # Exponential backoff
                    delay = min(2 ** attempt + (attempt * 0.1), 30)
                    await asyncio.sleep(delay)
                    continue
                    
            except Exception as e:
                last_exception = e
                
                # Check if this is the last attempt
                if attempt == self.config.max_retries:
                    metrics.complete(0, str(e))
                    self._request_metrics.append(metrics)
                    self._total_errors += 1
                    
                    self.logger.error(
                        f"Unexpected error [ID: {correlation_id}]: {e} (final attempt)",
                        extra={"correlation_id": correlation_id, "attempt": attempt + 1}
                    )
                    raise
                else:
                    # Log retry attempt for unexpected errors
                    self.logger.warning(
                        f"Unexpected error [ID: {correlation_id}]: {e}, retrying...",
                        extra={
                            "correlation_id": correlation_id,
                            "attempt": attempt + 1,
                            "retries_left": self.config.max_retries - attempt
                        }
                    )
                    
                    # Exponential backoff
                    delay = min(2 ** attempt + (attempt * 0.1), 30)
                    await asyncio.sleep(delay)
                    continue
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise GeminiAPIError("All retry attempts failed", status_code=0)
    
    def simple_message(self, content: str, role: str = "user") -> Dict[str, Any]:
        """Create a simple message dict with validation."""
        if not content.strip():
            raise GeminiValidationError("Message content cannot be empty")
        
        if role not in ["user", "model", "system"]:
            raise GeminiValidationError(f"Invalid role: {role}. Must be 'user', 'model', or 'system'")
        
        return {
            "role": role,
            "parts": [{"text": content}]
        }
    
    async def chat(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Send a chat message and get response text.
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system instructions
            chat_history: Optional previous messages
            correlation_id: Optional correlation ID for request tracing
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Response text from the model
        """
        if not prompt.strip():
            raise GeminiValidationError("Prompt cannot be empty")
        
        messages = []
        
        # Add system prompt if provided (Gemini doesn't support system role, prepend to first user message)
        if system_prompt:
            combined_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            prompt = combined_prompt
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
        
        # Add the current prompt
        messages.append(self.simple_message(prompt, role="user"))
        
        response = await self.generate_content(messages, correlation_id=correlation_id, **kwargs)
        
        try:
            response_text = response["candidates"][0]["content"]["parts"][0]["text"]
            return response_text
        except (KeyError, IndexError) as e:
            self.logger.error(
                f"Error parsing response structure: {e}",
                extra={
                    "correlation_id": correlation_id,
                    "response_keys": list(response.keys()) if isinstance(response, dict) else "not_dict"
                }
            )
            raise GeminiAPIError(f"Unexpected response structure: {e}", status_code=200)
    
    async def get_structured_response(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get a structured JSON response based on a schema with improved parsing.
        
        Args:
            prompt: User prompt
            output_schema: JSON schema for the response structure
            system_prompt: Optional system instructions
            correlation_id: Optional correlation ID for request tracing
            **kwargs: Additional parameters
            
        Returns:
            Structured response as a dictionary
        """
        if not output_schema:
            raise GeminiValidationError("Output schema cannot be empty")
        
        schema_prompt = f"""
        Please provide a response formatted according to the following JSON schema:
        
        {json.dumps(output_schema, indent=2)}
        
        Your response must be valid JSON that matches this schema exactly.
        Return ONLY the JSON, no additional text.
        """
        
        combined_prompt = f"{prompt}\n\n{schema_prompt}"
        
        # Enhanced system prompt for JSON output
        enhanced_system_prompt = (
            "You are a precise JSON generator. You must respond with valid JSON "
            "according to the provided schema. Do not include any text outside "
            "the JSON structure. Ensure all required fields are present."
        )
        
        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\n{enhanced_system_prompt}"
        
        response_text = await self.chat(
            combined_prompt, 
            system_prompt=enhanced_system_prompt,
            correlation_id=correlation_id,
            **kwargs
        )
        
        # Improved JSON extraction with multiple strategies
        try:
            # Strategy 1: Direct parsing
            return json.loads(response_text.strip())
            
        except json.JSONDecodeError:
            # Strategy 2: Extract JSON block from markdown
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Find JSON-like structure
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start_idx = response_text.find(start_char)
                end_idx = response_text.rfind(end_char)
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
            
            # Strategy 4: Clean and retry
            cleaned_text = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', response_text, flags=re.MULTILINE)
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                pass
            
            self.logger.error(
                f"Failed to parse JSON response: {response_text[:200]}...",
                extra={"correlation_id": correlation_id}
            )
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics."""
        if not self._request_metrics:
            return {
                "total_requests": 0,
                "total_errors": 0,
                "error_rate": 0.0,
                "average_duration": 0.0
            }
        
        successful_requests = [m for m in self._request_metrics if m.error is None]
        durations = [m.duration for m in self._request_metrics if m.duration is not None]
        
        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / self._total_requests if self._total_requests > 0 else 0.0,
            "average_duration": sum(durations) / len(durations) if durations else 0.0,
            "success_rate": len(successful_requests) / len(self._request_metrics),
            "recent_requests": len([m for m in self._request_metrics if time.time() - m.start_time < 3600])  # Last hour
        }


# Factory function for easy client creation
@asynccontextmanager
async def create_gemini_client(
    api_key: Optional[str] = None,
    **config_kwargs
) -> AsyncContextManager[GeminiClient]:
    """
    Factory function to create a Gemini client with async context manager.
    
    Usage:
        async with create_gemini_client() as client:
            result = await client.chat("Hello, world!")
    """
    config = ClientConfig(api_key=api_key, **config_kwargs)
    client = GeminiClient(config=config)
    try:
        yield client
    finally:
        await client.close()