"""Adaptive Provider Router: Intelligent LLM provider selection with failover and hedging."""

from __future__ import annotations
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class ProviderHealth(Enum):
    """Health status of a provider."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProviderStats:
    """Statistics for a provider."""
    provider_id: str
    health: ProviderHealth
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    total_requests: int = 0
    failed_requests: int = 0
    last_error: Optional[str] = None
    last_success: float = 0.0
    cost_per_token: float = 0.0


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    primary_provider: str
    hedged_providers: List[str]
    reason: str
    estimated_cost: float


class AdaptiveRouter:
    """
    Routes LLM requests to the best available provider.
    
    Features:
    - Health-based provider selection
    - Latency and reliability scoring
    - Request hedging for high-priority tasks
    - Automatic failover
    - Cost optimization
    """
    
    def __init__(self, providers: List[Dict[str, Any]], config=None):
        self.config = config or {}
        self.providers: Dict[str, ProviderStats] = {}
        self._lock = asyncio.Lock()
        
        # Initialize providers
        for prov in providers:
            provider_id = prov.get('id', prov.get('name'))
            self.providers[provider_id] = ProviderStats(
                provider_id=provider_id,
                health=ProviderHealth.UNKNOWN,
                cost_per_token=prov.get('cost_per_token', 0.0)
            )
        
        # Routing settings
        self.hedge_threshold_ms = config.get('hedge_threshold_ms', 2000)
        self.min_success_rate = config.get('min_success_rate', 0.8)
        self.health_check_interval = config.get('health_check_interval', 60)
        
        # Start background health monitoring
        self._health_monitor_task: Optional[asyncio.Task] = None
    
    def start(self):
        """Start the background health monitor."""
        if self._health_monitor_task is None:
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
    
    def stop(self):
        """Stop the health monitor."""
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
    
    async def _health_monitor_loop(self):
        """Background loop to check provider health."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_providers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health monitor error: {e}")
    
    async def _check_all_providers(self):
        """Check health of all providers."""
        # In a real implementation, this would send test requests
        # For now, we'll just age the health status
        now = time.time()
        
        async with self._lock:
            for provider in self.providers.values():
                if provider.last_success > 0:
                    time_since_success = now - provider.last_success
                    
                    # Degrade health if no recent success
                    if time_since_success > 300:  # 5 minutes
                        provider.health = ProviderHealth.DEGRADED
                    if time_since_success > 600:  # 10 minutes
                        provider.health = ProviderHealth.UNHEALTHY
    
    async def generate(
        self, 
        prompt: str, 
        priority: str = "normal",
        hedge: bool = False,
        **kwargs
    ) -> Any:
        """
        Generate a response using the best available provider.
        
        Args:
            prompt: The prompt to send
            priority: Priority level ("low", "normal", "high", "critical")
            hedge: Whether to use request hedging
            **kwargs: Additional arguments for the LLM
            
        Returns:
            LLM response
        """
        # Select providers
        decision = self._select_providers(priority, hedge)
        
        if not decision.primary_provider:
            raise RuntimeError("No healthy providers available")
        
        # Execute request
        return await self._execute_with_failover(decision, prompt, **kwargs)
    
    def _select_providers(self, priority: str, hedge: bool) -> RoutingDecision:
        """
        Select primary and hedged providers based on current state.
        
        Returns:
            RoutingDecision with selected providers
        """
        healthy_providers = []
        
        for provider_id, stats in self.providers.items():
            if (stats.health in [ProviderHealth.HEALTHY, ProviderHealth.DEGRADED] and
                stats.success_rate >= self.min_success_rate):
                
                # Calculate score (higher is better)
                score = self._calculate_score(stats, priority)
                healthy_providers.append((provider_id, score, stats))
        
        if not healthy_providers:
            return RoutingDecision(
                primary_provider="",
                hedged_providers=[],
                reason="No healthy providers",
                estimated_cost=0.0
            )
        
        # Sort by score (descending)
        healthy_providers.sort(key=lambda x: x[1], reverse=True)
        
        primary_id = healthy_providers[0][0]
        primary_stats = healthy_providers[0][2]
        
        # Decide on hedging
        hedged = []
        if hedge or (priority in ["high", "critical"] and primary_stats.avg_latency_ms > self.hedge_threshold_ms):
            # Select 1-2 backup providers
            for prov_id, _, stats in healthy_providers[1:3]:
                hedged.append(prov_id)
        
        return RoutingDecision(
            primary_provider=primary_id,
            hedged_providers=hedged,
            reason=f"Selected based on score and health",
            estimated_cost=primary_stats.cost_per_token * 100  # Estimate
        )
    
    def _calculate_score(self, stats: ProviderStats, priority: str) -> float:
        """
        Calculate a routing score for a provider.
        
        Higher score = better choice
        """
        # Base score from success rate (0-1)
        base_score = stats.success_rate
        
        # Latency penalty (normalize to 0-1, lower latency = higher score)
        latency_score = max(0, 1 - (stats.avg_latency_ms / 5000))
        
        # Health bonus
        health_bonus = {
            ProviderHealth.HEALTHY: 0.2,
            ProviderHealth.DEGRADED: 0.1,
            ProviderHealth.UNHEALTHY: 0.0,
            ProviderHealth.UNKNOWN: 0.0
        }.get(stats.health, 0.0)
        
        # Cost factor (for low priority, prefer cheaper)
        cost_factor = 1.0
        if priority == "low":
            cost_factor = 1 - (stats.cost_per_token * 10)  # Prefer cheaper
        
        # Priority weighting
        if priority in ["high", "critical"]:
            # Prioritize reliability and speed over cost
            score = (base_score * 0.4 + latency_score * 0.4 + health_bonus) * cost_factor
        else:
            # Balanced approach
            score = (base_score * 0.3 + latency_score * 0.3 + health_bonus + cost_factor * 0.1)
        
        return score
    
    async def _execute_with_failover(
        self, 
        decision: RoutingDecision, 
        prompt: str,
        **kwargs
    ) -> Any:
        """
        Execute request with automatic failover.
        
        Args:
            decision: Routing decision with providers
            prompt: The prompt to send
            **kwargs: Additional LLM arguments
            
        Returns:
            LLM response
        """
        all_providers = [decision.primary_provider] + decision.hedged_providers
        
        for i, provider_id in enumerate(all_providers):
            try:
                # Record request start
                start_time = time.time()
                
                # Execute request
                result = await self._call_provider(provider_id, prompt, **kwargs)
                
                # Update stats on success
                latency_ms = (time.time() - start_time) * 1000
                await self._record_success(provider_id, latency_ms)
                
                return result
                
            except Exception as e:
                # Record failure
                await self._record_failure(provider_id, str(e))
                
                # Try next provider if available
                if i < len(all_providers) - 1:
                    print(f"Provider {provider_id} failed, trying next...")
                    continue
                else:
                    raise RuntimeError(f"All providers failed. Last error: {e}")
        
        raise RuntimeError("No providers available")
    
    async def _call_provider(self, provider_id: str, prompt: str, **kwargs) -> Any:
        """
        Call a specific provider.
        
        This is a placeholder - in reality would call actual LLM APIs.
        """
        # Simulate provider call
        await asyncio.sleep(0.1)  # Simulate network latency
        
        # Return mock response
        return type('obj', (object,), {
            'content': f"Response from {provider_id}",
            'provider': provider_id,
            'token_count': 50
        })
    
    async def _record_success(self, provider_id: str, latency_ms: float):
        """Record a successful request."""
        async with self._lock:
            if provider_id not in self.providers:
                return
            
            stats = self.providers[provider_id]
            stats.total_requests += 1
            stats.last_success = time.time()
            stats.health = ProviderHealth.HEALTHY
            
            # Update rolling average latency
            n = stats.total_requests
            stats.avg_latency_ms = ((n - 1) * stats.avg_latency_ms + latency_ms) / n
    
    async def _record_failure(self, provider_id: str, error: str):
        """Record a failed request."""
        async with self._lock:
            if provider_id not in self.providers:
                return
            
            stats = self.providers[provider_id]
            stats.total_requests += 1
            stats.failed_requests += 1
            stats.last_error = error
            
            # Update success rate
            stats.success_rate = 1 - (stats.failed_requests / stats.total_requests)
            
            # Degrade health on repeated failures
            if stats.success_rate < 0.5:
                stats.health = ProviderHealth.UNHEALTHY
            elif stats.success_rate < 0.8:
                stats.health = ProviderHealth.DEGRADED
    
    def get_provider_stats(self, provider_id: str) -> Optional[ProviderStats]:
        """Get statistics for a specific provider."""
        return self.providers.get(provider_id)
    
    def list_healthy_providers(self) -> List[str]:
        """List all currently healthy providers."""
        return [
            pid for pid, stats in self.providers.items()
            if stats.health in [ProviderHealth.HEALTHY, ProviderHealth.DEGRADED]
        ]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get overall routing statistics."""
        total_requests = sum(s.total_requests for s in self.providers.values())
        total_failures = sum(s.failed_requests for s in self.providers.values())
        
        return {
            'total_providers': len(self.providers),
            'healthy_providers': len(self.list_healthy_providers()),
            'total_requests': total_requests,
            'total_failures': total_failures,
            'overall_success_rate': 1 - (total_failures / total_requests) if total_requests > 0 else 0,
            'providers': {
                pid: {
                    'health': stats.health.value,
                    'success_rate': stats.success_rate,
                    'avg_latency_ms': stats.avg_latency_ms
                }
                for pid, stats in self.providers.items()
            }
        }
