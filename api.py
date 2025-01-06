# api.py
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import uvicorn
import logging
import json
from functools import lru_cache
import time
from cachetools import TTLCache
import hashlib
import uuid

# Import our feedback agent (feedback_agent.py)
from feedback_agent import run_feedback_analysis, FeedbackCategory, Priority, TaskType, Task




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Request/Response Models
class FeedbackRequest(BaseModel):
    feedback_text: str
    customer_id: str = None
    timestamp: datetime = datetime.now()

class FeedbackResponse(BaseModel):
    feedback: str
    sentiment: str
    category: FeedbackCategory
    priority: Priority
    next_steps: List[Task]
    customer_id: str = None
    timestamp: datetime
    request_id: str
    cache_hit: bool = False

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime
    request_id: str

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
valid_keys = set()
try:
    with open('hapi.txt', 'r') as keyfile:
        valid_keys = {key.strip() for key in keyfile.readlines()}
except FileNotFoundError:
    valid_keys.add("test")  # Default key for testing
    logger.info(f"Configuration Error: API key validation file not found. Using default key for testing.")

# Cache configuration
response_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache responses for 1 hour

# Custom exception for API errors
class APIError(Exception):
    def __init__(self, error: str, detail: str, status_code: int = 400):
        self.error = error
        self.detail = detail
        self.status_code = status_code
        super().__init__(self.detail)

# API Key validation
async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header not in valid_keys:  # Replace with your secure key management
        raise APIError(
            error="Invalid API Key",
            detail="The provided API key is not valid",
            status_code=401
        )
    return api_key_header

# Create FastAPI app
app = FastAPI(
    title="Customer Feedback AI Agent API",
    description="API for analyzing customer feedback using AI",
    version="1.0.0"
)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Log request
    logger.info(f"Request {request_id} started: {request.method} {request.url}")

    try:
        # Get request body
        body = await request.body()
        if body:
            logger.debug(f"Request body: {body.decode()}")

        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        logger.info(
            f"Request {request_id} completed: Status {response.status_code} "
            f"Process Time: {process_time:.2f}s"
        )

        return response

    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise

# Error handler
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    error_response = ErrorResponse(
        error=exc.error,
        detail=exc.detail,
        timestamp=datetime.now(),
        request_id=str(uuid.uuid4())
    )
    logger.error(f"API Error: {error_response.json()}")
    return Response(
        content=error_response.json(),
        status_code=exc.status_code,
        media_type="application/json"
    )

# Cache key generator
def generate_cache_key(feedback_text: str, customer_id: Optional[str] = None) -> str:
    """Generate a cache key from the feedback text and customer ID"""
    key_string = f"{feedback_text}:{customer_id}"
    return hashlib.md5(key_string.encode()).hexdigest()

# Health check endpoint
@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "cache_size": len(response_cache),
        "cache_maxsize": response_cache.maxsize
    }

# Main feedback analysis endpoint
@app.post("/analyze-feedback", response_model=FeedbackResponse)
async def analyze_feedback(
    request: FeedbackRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Generate cache key
        cache_key = generate_cache_key(request.feedback_text, request.customer_id)

        # Check cache
        cached_response = response_cache.get(cache_key)
        if cached_response:
            logger.info(f"Cache hit for key: {cache_key}")
            cached_response["cache_hit"] = True
            return FeedbackResponse(**cached_response)

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Run the feedback analysis
        results = run_feedback_analysis(request.feedback_text)

        # Prepare response
        response_data = {
            "feedback": request.feedback_text,
            "sentiment": results["sentiment"],
            "category": results["category"],
            "priority": results["priority"],
            "next_steps": results["next_steps"],
            "customer_id": request.customer_id,
            "timestamp": request.timestamp or datetime.now(),
            "request_id": request_id,
            "cache_hit": False
        }

        # Cache the response
        response_cache[cache_key] = response_data

        return FeedbackResponse(**response_data)

    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        raise APIError(
            error="Processing Error",
            detail=str(e),
            status_code=500
        )

# Batch feedback analysis endpoint
@app.post("/analyze-feedback-batch")
async def analyze_feedback_batch(
    requests: List[FeedbackRequest],
    api_key: str = Depends(get_api_key)
):
    try:
        results = []
        for request in requests:
            # Check cache first
            cache_key = generate_cache_key(request.feedback_text, request.customer_id)
            cached_response = response_cache.get(cache_key)

            if cached_response:
                cached_response["cache_hit"] = True
                results.append(FeedbackResponse(**cached_response))
                continue

            # Process new feedback
            request_id = str(uuid.uuid4())
            analysis = run_feedback_analysis(request.feedback_text)

            response_data = {
                "feedback": request.feedback_text,
                "sentiment": analysis["sentiment"],
                "category": analysis["category"],
                "priority": analysis["priority"],
                "next_steps": analysis["next_steps"],
                "customer_id": request.customer_id,
                "timestamp": request.timestamp or datetime.now(),
                "request_id": request_id,
                "cache_hit": False
            }

            # Cache the response
            response_cache[cache_key] = response_data
            results.append(FeedbackResponse(**response_data))

        return results

    except Exception as e:
        logger.error(f"Error processing batch feedback: {str(e)}", exc_info=True)
        raise APIError(
            error="Batch Processing Error",
            detail=str(e),
            status_code=500
        )

# Get feedback categories endpoint
@app.get("/categories")
async def get_categories(api_key: str = Depends(get_api_key)):
    return {"categories": [category.value for category in FeedbackCategory]}

# Get priority levels endpoint
@app.get("/priorities")
async def get_priorities(api_key: str = Depends(get_api_key)):
    return {"priorities": [priority.value for priority in Priority]}

# Cache management endpoints
@app.post("/cache/clear")
async def clear_cache(api_key: str = Depends(get_api_key)):
    response_cache.clear()
    return {"message": "Cache cleared", "timestamp": datetime.now()}

@app.get("/cache/stats")
async def get_cache_stats(api_key: str = Depends(get_api_key)):
    return {
        "size": len(response_cache),
        "maxsize": response_cache.maxsize,
        "ttl": response_cache.ttl,
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)