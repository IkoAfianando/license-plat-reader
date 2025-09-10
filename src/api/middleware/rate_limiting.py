"""
Rate Limiting Middleware for License Plate Reader API
Implements token bucket and sliding window rate limiting
"""

import time
import json
import hashlib
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class RateLimitRule:
    """Rate limiting rule configuration"""
    
    def __init__(self, 
                 requests: int,
                 window_seconds: int,
                 burst: int = None,
                 key_func: callable = None):
        """
        Initialize rate limit rule
        
        Args:
            requests: Number of requests allowed
            window_seconds: Time window in seconds
            burst: Maximum burst requests (for token bucket)
            key_func: Function to generate rate limit key from request
        """
        self.requests = requests
        self.window_seconds = window_seconds
        self.burst = burst or requests
        self.key_func = key_func or self._default_key_func
    
    def _default_key_func(self, request: Request) -> str:
        """Default key function using client IP"""
        client_ip = self._get_client_ip(request)
        return f"rate_limit:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers (from load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client
        return request.client.host if request.client else "unknown"

class InMemoryRateLimiter:
    """In-memory rate limiter using sliding window"""
    
    def __init__(self):
        self.requests = defaultdict(deque)  # Store request timestamps
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, rule: RateLimitRule) -> Tuple[bool, Dict]:
        """
        Check if request is allowed under rate limit
        
        Returns:
            Tuple of (allowed, info_dict)
        """
        async with self.lock:
            current_time = time.time()
            window_start = current_time - rule.window_seconds
            
            # Clean old requests outside the window
            request_times = self.requests[key]
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            
            # Check if we're under the limit
            current_requests = len(request_times)
            allowed = current_requests < rule.requests
            
            # If allowed, record this request
            if allowed:
                request_times.append(current_time)
            
            # Calculate reset time
            if request_times:
                reset_time = request_times[0] + rule.window_seconds
            else:
                reset_time = current_time + rule.window_seconds
            
            info = {
                'requests_made': current_requests + (1 if allowed else 0),
                'requests_limit': rule.requests,
                'reset_time': reset_time,
                'window_seconds': rule.window_seconds,
                'retry_after': max(0, reset_time - current_time) if not allowed else 0
            }
            
            return allowed, info

class RedisRateLimiter:
    """Redis-backed rate limiter for distributed systems"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def is_allowed(self, key: str, rule: RateLimitRule) -> Tuple[bool, Dict]:
        """
        Check if request is allowed using Redis sliding window
        
        Returns:
            Tuple of (allowed, info_dict)
        """
        current_time = time.time()
        window_start = current_time - rule.window_seconds
        
        pipeline = self.redis.pipeline()
        
        try:
            # Remove old entries
            pipeline.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            pipeline.zcard(key)
            
            # Execute pipeline
            results = await pipeline.execute()
            current_requests = results[1]
            
            allowed = current_requests < rule.requests
            
            if allowed:
                # Add current request
                pipeline = self.redis.pipeline()
                pipeline.zadd(key, {str(current_time): current_time})
                pipeline.expire(key, rule.window_seconds)
                await pipeline.execute()
            
            # Get oldest entry for reset time calculation
            oldest_entry = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest_entry:
                reset_time = oldest_entry[0][1] + rule.window_seconds
            else:
                reset_time = current_time + rule.window_seconds
            
            info = {
                'requests_made': current_requests + (1 if allowed else 0),
                'requests_limit': rule.requests,
                'reset_time': reset_time,
                'window_seconds': rule.window_seconds,
                'retry_after': max(0, reset_time - current_time) if not allowed else 0
            }
            
            return allowed, info
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fallback to allowing the request if Redis fails
            return True, {
                'requests_made': 0,
                'requests_limit': rule.requests,
                'reset_time': current_time + rule.window_seconds,
                'window_seconds': rule.window_seconds,
                'retry_after': 0,
                'error': 'Redis unavailable'
            }

class TokenBucketRateLimiter:
    """Token bucket rate limiter for handling bursts"""
    
    def __init__(self):
        self.buckets = defaultdict(dict)  # Store bucket state
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, rule: RateLimitRule) -> Tuple[bool, Dict]:
        """
        Check if request is allowed using token bucket algorithm
        
        Returns:
            Tuple of (allowed, info_dict)
        """
        async with self.lock:
            current_time = time.time()
            
            # Get or initialize bucket
            bucket = self.buckets[key]
            if not bucket:
                bucket.update({
                    'tokens': rule.burst,
                    'last_refill': current_time
                })
            
            # Calculate tokens to add based on time elapsed
            time_elapsed = current_time - bucket['last_refill']
            tokens_to_add = time_elapsed * (rule.requests / rule.window_seconds)
            
            # Add tokens, but don't exceed bucket capacity
            bucket['tokens'] = min(rule.burst, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = current_time
            
            # Check if we have tokens available
            allowed = bucket['tokens'] >= 1
            
            if allowed:
                bucket['tokens'] -= 1
            
            # Calculate when next token will be available
            if bucket['tokens'] < 1:
                retry_after = (1 - bucket['tokens']) / (rule.requests / rule.window_seconds)
            else:
                retry_after = 0
            
            info = {
                'tokens_remaining': max(0, bucket['tokens']),
                'bucket_capacity': rule.burst,
                'refill_rate': rule.requests / rule.window_seconds,
                'retry_after': retry_after
            }
            
            return allowed, info

class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, 
                 redis_client=None,
                 use_token_bucket: bool = False,
                 default_rules: Dict[str, RateLimitRule] = None):
        """
        Initialize rate limiting middleware
        
        Args:
            redis_client: Redis client for distributed rate limiting
            use_token_bucket: Use token bucket algorithm instead of sliding window
            default_rules: Default rate limiting rules by endpoint pattern
        """
        self.redis_client = redis_client
        self.use_token_bucket = use_token_bucket
        
        # Choose rate limiter based on availability and preference
        if redis_client and REDIS_AVAILABLE and not use_token_bucket:
            self.limiter = RedisRateLimiter(redis_client)
            logger.info("Using Redis-based rate limiter")
        elif use_token_bucket:
            self.limiter = TokenBucketRateLimiter()
            logger.info("Using token bucket rate limiter")
        else:
            self.limiter = InMemoryRateLimiter()
            logger.info("Using in-memory sliding window rate limiter")
        
        # Default rate limiting rules
        self.rules = default_rules or {
            "/detect/image": RateLimitRule(10, 60),      # 10 requests per minute
            "/detect/batch": RateLimitRule(5, 60),       # 5 requests per minute
            "/detect/video": RateLimitRule(2, 300),      # 2 requests per 5 minutes
            "/auth/login": RateLimitRule(5, 300),        # 5 login attempts per 5 minutes
            "default": RateLimitRule(100, 60)            # 100 requests per minute default
        }
        
        logger.info(f"Rate limiting initialized with {len(self.rules)} rules")
    
    def add_rule(self, endpoint: str, rule: RateLimitRule):
        """Add or update rate limiting rule for endpoint"""
        self.rules[endpoint] = rule
        logger.info(f"Added rate limit rule for {endpoint}: {rule.requests} requests per {rule.window_seconds}s")
    
    def get_rule_for_request(self, request: Request) -> RateLimitRule:
        """Get the appropriate rate limiting rule for a request"""
        path = request.url.path
        
        # Exact match first
        if path in self.rules:
            return self.rules[path]
        
        # Pattern matching
        for pattern, rule in self.rules.items():
            if pattern != "default" and pattern in path:
                return rule
        
        # Default rule
        return self.rules.get("default", RateLimitRule(100, 60))
    
    async def __call__(self, request: Request, call_next):
        """Middleware function to check rate limits"""
        
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        # Get rate limiting rule
        rule = self.get_rule_for_request(request)
        
        # Generate rate limiting key
        rate_limit_key = rule.key_func(request)
        
        # Check rate limit
        allowed, info = await self.limiter.is_allowed(rate_limit_key, rule)
        
        if not allowed:
            # Rate limit exceeded
            headers = {
                "X-RateLimit-Limit": str(info.get('requests_limit', rule.requests)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(info.get('reset_time', time.time() + rule.window_seconds))),
                "Retry-After": str(int(info.get('retry_after', rule.window_seconds)))
            }
            
            error_response = {
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Limit: {rule.requests} per {rule.window_seconds} seconds",
                "retry_after": int(info.get('retry_after', rule.window_seconds)),
                "limit_info": info
            }
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error_response,
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        remaining = info.get('requests_limit', rule.requests) - info.get('requests_made', 1)
        response.headers.update({
            "X-RateLimit-Limit": str(info.get('requests_limit', rule.requests)),
            "X-RateLimit-Remaining": str(max(0, remaining)),
            "X-RateLimit-Reset": str(int(info.get('reset_time', time.time() + rule.window_seconds)))
        })
        
        return response

class AuthenticatedRateLimitRule(RateLimitRule):
    """Rate limiting rule that uses authenticated user ID instead of IP"""
    
    def __init__(self, requests: int, window_seconds: int, burst: int = None):
        super().__init__(requests, window_seconds, burst, self._user_key_func)
    
    def _user_key_func(self, request: Request) -> str:
        """Generate key based on authenticated user"""
        # Extract user from JWT token in Authorization header
        auth_header = request.headers.get("Authorization", "")
        
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                import jwt
                # This should match the JWT secret in your API
                payload = jwt.decode(token, verify=False)  # Just decode without verification for key generation
                user_id = payload.get("username", "anonymous")
                return f"rate_limit:user:{user_id}"
            except:
                pass
        
        # Fallback to IP-based limiting
        client_ip = self._get_client_ip(request)
        return f"rate_limit:ip:{client_ip}"

class EndpointSpecificRateLimitRule(RateLimitRule):
    """Rate limiting rule that includes endpoint in the key"""
    
    def __init__(self, requests: int, window_seconds: int, burst: int = None):
        super().__init__(requests, window_seconds, burst, self._endpoint_key_func)
    
    def _endpoint_key_func(self, request: Request) -> str:
        """Generate key based on user and endpoint"""
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path
        
        # Create hash of endpoint to keep key length reasonable
        endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
        
        return f"rate_limit:{client_ip}:{endpoint_hash}"

def create_detection_rate_limiter(redis_client=None) -> RateLimitMiddleware:
    """Create rate limiter specifically configured for detection API"""
    
    # Define rules for different detection endpoints
    rules = {
        # Single image detection - moderate limits
        "/detect/image": AuthenticatedRateLimitRule(20, 60),  # 20 per minute per user
        
        # Batch processing - stricter limits
        "/detect/batch": AuthenticatedRateLimitRule(5, 300),  # 5 per 5 minutes per user
        
        # Video processing - very strict limits
        "/detect/video": AuthenticatedRateLimitRule(2, 600),  # 2 per 10 minutes per user
        
        # Authentication - prevent brute force
        "/auth/login": EndpointSpecificRateLimitRule(5, 900),  # 5 attempts per 15 minutes per IP
        
        # Data upload - moderate limits
        "/data/upload": AuthenticatedRateLimitRule(10, 300),  # 10 per 5 minutes per user
        
        # Analytics - generous limits
        "/analytics/stats": AuthenticatedRateLimitRule(60, 60),  # 60 per minute per user
        
        # General API - default limits
        "default": RateLimitRule(100, 60)  # 100 per minute per IP
    }
    
    return RateLimitMiddleware(
        redis_client=redis_client,
        use_token_bucket=False,
        default_rules=rules
    )

def main():
    """Demo/test function for rate limiting"""
    import asyncio
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    
    print("🚦 Rate Limiting Middleware Demo")
    print("=" * 40)
    
    # Create demo rate limiter
    rate_limiter = RateLimitMiddleware(
        default_rules={
            "/test": RateLimitRule(3, 10),  # 3 requests per 10 seconds
            "default": RateLimitRule(5, 10)  # 5 requests per 10 seconds
        }
    )
    
    # Create mock FastAPI app
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)
    
    # Create mock request for testing
    class MockRequest:
        def __init__(self, path="/test"):
            self.url = type('obj', (object,), {'path': path})
            self.client = type('obj', (object,), {'host': '127.0.0.1'})
            self.headers = {}
    
    async def test_rate_limiting():
        """Test rate limiting functionality"""
        print("\\n🧪 Testing rate limiting...")
        
        # Test multiple requests
        for i in range(6):
            request = MockRequest("/test")
            rule = rate_limiter.get_rule_for_request(request)
            
            allowed, info = await rate_limiter.limiter.is_allowed(
                rule.key_func(request), 
                rule
            )
            
            status = "✅ ALLOWED" if allowed else "❌ BLOCKED"
            print(f"Request {i+1}: {status}")
            print(f"  Requests made: {info['requests_made']}/{info['requests_limit']}")
            print(f"  Retry after: {info['retry_after']:.1f}s")
            
            if not allowed:
                break
            
            await asyncio.sleep(0.1)
        
        print("\\n✅ Rate limiting test completed!")
    
    # Run test
    asyncio.run(test_rate_limiting())

if __name__ == "__main__":
    main()