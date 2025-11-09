"""
Redis caching for inference results with disk fallback.
"""

import hashlib
import pickle
import json
from typing import Optional, Any, Tuple
from pathlib import Path
import numpy as np


class CacheClient:
    """
    Cache client with Redis primary and disk fallback.
    """
    
    def __init__(self, redis_url: Optional[str] = None, cache_dir: str = "cache",
                 ttl: int = 604800):  # 7 days default TTL
        self.redis_client = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        
        self.stats = {"hits": 0, "misses": 0, "errors": 0}
        
        # Try to connect to Redis
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                self.redis_client.ping()
                print(f"Connected to Redis at {redis_url}")
            except Exception as e:
                print(f"Warning: Could not connect to Redis: {e}")
                print("Falling back to disk cache only")
                self.redis_client = None
        else:
            print("No Redis URL provided, using disk cache only")
    
    def _generate_key(self, image_bytes: bytes, method: str, params: dict) -> str:
        """Generate cache key from image and parameters."""
        # Create deterministic hash
        hasher = hashlib.sha256()
        hasher.update(image_bytes)
        hasher.update(method.encode())
        hasher.update(json.dumps(params, sort_keys=True).encode())
        return hasher.hexdigest()
    
    def get(self, image_bytes: bytes, method: str, params: dict) -> Optional[np.ndarray]:
        """
        Get cached result.
        
        Returns:
            Cached image as numpy array or None if not found
        """
        key = self._generate_key(image_bytes, method, params)
        
        # Try Redis first
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    self.stats["hits"] += 1
                    result = pickle.loads(data)
                    return result
            except Exception as e:
                print(f"Redis get error: {e}")
                self.stats["errors"] += 1
        
        # Try disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                self.stats["hits"] += 1
                
                # Optionally sync back to Redis if available
                if self.redis_client and result is not None:
                    try:
                        self.redis_client.setex(key, self.ttl, pickle.dumps(result))
                    except Exception as e:
                        # Silent fail for optional Redis sync - disk cache is primary
                        self.stats["errors"] += 1
                
                return result
            except Exception as e:
                print(f"Disk cache read error: {e}")
                self.stats["errors"] += 1
        
        self.stats["misses"] += 1
        return None
    
    def set(self, image_bytes: bytes, method: str, params: dict, result: np.ndarray):
        """Cache result."""
        key = self._generate_key(image_bytes, method, params)
        data = pickle.dumps(result)
        
        # Save to Redis
        if self.redis_client:
            try:
                self.redis_client.setex(key, self.ttl, data)
            except Exception as e:
                print(f"Redis set error: {e}")
                self.stats["errors"] += 1
        
        # Save to disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(f"Disk cache write error: {e}")
            self.stats["errors"] += 1
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0.0
        
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate": hit_rate
        }
    
    def flush(self):
        """Clear all cache."""
        # Clear Redis
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                print("Redis cache flushed")
            except Exception as e:
                print(f"Redis flush error: {e}")
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"Error deleting {cache_file}: {e}")
        
        print("Disk cache flushed")
        self.stats = {"hits": 0, "misses": 0, "errors": 0}
    
    def close(self):
        """Close connections."""
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception:
                # Redis client may already be closed or disconnected
                # This is safe to ignore during cleanup
                self.stats["errors"] += 1


# Global cache instance
_cache_instance: Optional[CacheClient] = None


def get_cache(redis_url: Optional[str] = None, cache_dir: str = "cache") -> CacheClient:
    """Get or create global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheClient(redis_url=redis_url, cache_dir=cache_dir)
    return _cache_instance
