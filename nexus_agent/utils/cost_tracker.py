"""Cost Tracker: Real-time budget enforcement and usage tracking."""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded."""
    pass


@dataclass
class TaskUsage:
    """Usage statistics for a single task."""
    task_id: str
    start_time: float
    end_time: Optional[float] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    provider: Optional[str] = None
    status: str = "running"


@dataclass
class BudgetConfig:
    """Budget configuration."""
    hourly_limit_usd: float = 10.0
    daily_limit_usd: float = 100.0
    per_task_limit_usd: float = 5.0
    token_limit_per_hour: int = 100000


class CostTracker:
    """
    Tracks and enforces budget limits for LLM usage.
    
    Features:
    - Hourly/daily/per-task budget limits
    - Real-time cost calculation
    - Token usage tracking
    - Budget alerts
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = BudgetConfig(**config) if config else BudgetConfig()
        
        # Active tasks
        self.active_tasks: Dict[str, TaskUsage] = {}
        
        # Historical usage
        self.task_history: List[TaskUsage] = []
        
        # Time-based tracking
        self.hourly_usage: Dict[int, float] = {}  # hour -> cost
        self.daily_usage: Dict[int, float] = {}   # day -> cost
        
        # Default cost per token (can be overridden per-provider)
        self.cost_per_token = 0.00002  # $0.02 per 1K tokens
    
    def start_task(self, task_id: str, budget_override: Optional[float] = None) -> bool:
        """
        Start tracking a new task. Checks budgets before allowing.
        
        Args:
            task_id: Unique task identifier
            budget_override: Optional override for per-task limit
            
        Returns:
            True if task can proceed
            
        Raises:
            BudgetExceededError: If any budget limit is exceeded
        """
        # Check hourly budget
        current_hour = int(time.time() // 3600)
        hourly_spent = self.hourly_usage.get(current_hour, 0)
        
        if hourly_spent >= self.config.hourly_limit_usd:
            raise BudgetExceededError(
                f"Hourly budget exceeded: ${hourly_spent:.2f} / ${self.config.hourly_limit_usd:.2f}"
            )
        
        # Check daily budget
        current_day = int(time.time() // 86400)
        daily_spent = self.daily_usage.get(current_day, 0)
        
        if daily_spent >= self.config.daily_limit_usd:
            raise BudgetExceededError(
                f"Daily budget exceeded: ${daily_spent:.2f} / ${self.config.daily_limit_usd:.2f}"
            )
        
        # Create task usage record
        task_limit = budget_override or self.config.per_task_limit_usd
        
        usage = TaskUsage(
            task_id=task_id,
            start_time=time.time(),
            status="running"
        )
        
        self.active_tasks[task_id] = usage
        return True
    
    def record_tokens(
        self, 
        task_id: str, 
        tokens: int, 
        provider: Optional[str] = None,
        cost_override: Optional[float] = None
    ):
        """
        Record token usage for a task.
        
        Args:
            task_id: Task identifier
            tokens: Number of tokens used
            provider: LLM provider name
            cost_override: Optional cost override
        """
        if task_id not in self.active_tasks:
            return
        
        usage = self.active_tasks[task_id]
        usage.tokens_used += tokens
        usage.provider = provider
        
        # Calculate cost
        if cost_override is not None:
            task_cost = cost_override
        else:
            task_cost = tokens * self.cost_per_token
        
        usage.cost_usd += task_cost
        
        # Update time-based tracking
        current_hour = int(time.time() // 3600)
        current_day = int(time.time() // 86400)
        
        self.hourly_usage[current_hour] = self.hourly_usage.get(current_hour, 0) + task_cost
        self.daily_usage[current_day] = self.daily_usage.get(current_day, 0) + task_cost
    
    def end_task(self, task_id: str) -> Dict[str, Any]:
        """
        End a task and return final usage summary.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Usage summary dictionary
        """
        if task_id not in self.active_tasks:
            return {"error": "Task not found"}
        
        usage = self.active_tasks.pop(task_id)
        usage.end_time = time.time()
        usage.status = "completed"
        
        # Add to history
        self.task_history.append(usage)
        
        # Keep history bounded
        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-1000:]
        
        return {
            "task_id": task_id,
            "tokens_used": usage.tokens_used,
            "cost_usd": usage.cost_usd,
            "duration_seconds": usage.end_time - usage.start_time,
            "provider": usage.provider
        }
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        current_hour = int(time.time() // 3600)
        current_day = int(time.time() // 86400)
        
        hourly_spent = self.hourly_usage.get(current_hour, 0)
        daily_spent = self.daily_usage.get(current_day, 0)
        
        active_task_count = len(self.active_tasks)
        active_task_cost = sum(t.cost_usd for t in self.active_tasks.values())
        
        return {
            "hourly": {
                "spent": hourly_spent,
                "limit": self.config.hourly_limit_usd,
                "remaining": self.config.hourly_limit_usd - hourly_spent,
                "percent_used": (hourly_spent / self.config.hourly_limit_usd) * 100
            },
            "daily": {
                "spent": daily_spent,
                "limit": self.config.daily_limit_usd,
                "remaining": self.config.daily_limit_usd - daily_spent,
                "percent_used": (daily_spent / self.config.daily_limit_usd) * 100
            },
            "active_tasks": {
                "count": active_task_count,
                "estimated_cost": active_task_cost
            },
            "total_tasks_completed": len(self.task_history)
        }
    
    def get_task_usage(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get usage for a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            usage = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "running",
                "tokens_used": usage.tokens_used,
                "cost_usd": usage.cost_usd,
                "elapsed_seconds": time.time() - usage.start_time
            }
        
        # Check history
        for usage in self.task_history:
            if usage.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": usage.status,
                    "tokens_used": usage.tokens_used,
                    "cost_usd": usage.cost_usd,
                    "duration_seconds": (usage.end_time - usage.start_time) if usage.end_time else None
                }
        
        return None
    
    def reset_hourly(self):
        """Reset hourly tracking (called automatically based on time)."""
        current_hour = int(time.time() // 3600)
        
        # Remove old hours
        old_hours = [h for h in self.hourly_usage if h < current_hour]
        for h in old_hours:
            del self.hourly_usage[h]
    
    def reset_daily(self):
        """Reset daily tracking (called automatically based on time)."""
        current_day = int(time.time() // 86400)
        
        # Remove old days
        old_days = [d for d in self.daily_usage if d < current_day]
        for d in old_days:
            del self.daily_usage[d]
    
    def set_provider_cost(self, provider: str, cost_per_token: float):
        """Set cost per token for a specific provider."""
        # In a real implementation, would maintain a provider cost map
        # For now, just update the global rate
        self.cost_per_token = cost_per_token
    
    def get_budget_alerts(self) -> List[Dict[str, Any]]:
        """Get list of budget alerts (threshold warnings)."""
        alerts = []
        
        current_hour = int(time.time() // 3600)
        current_day = int(time.time() // 86400)
        
        hourly_percent = (self.hourly_usage.get(current_hour, 0) / self.config.hourly_limit_usd) * 100
        daily_percent = (self.daily_usage.get(current_day, 0) / self.config.daily_limit_usd) * 100
        
        if hourly_percent >= 90:
            alerts.append({
                "type": "hourly_warning",
                "message": f"Hourly budget at {hourly_percent:.1f}%",
                "severity": "critical" if hourly_percent >= 100 else "warning"
            })
        
        if daily_percent >= 90:
            alerts.append({
                "type": "daily_warning",
                "message": f"Daily budget at {daily_percent:.1f}%",
                "severity": "critical" if daily_percent >= 100 else "warning"
            })
        
        return alerts
    
    def export_usage_report(self) -> Dict[str, Any]:
        """Export a comprehensive usage report."""
        return {
            "generated_at": time.time(),
            "current_usage": self.get_current_usage(),
            "alerts": self.get_budget_alerts(),
            "recent_tasks": [
                {
                    "task_id": t.task_id,
                    "tokens": t.tokens_used,
                    "cost": t.cost_usd,
                    "duration": (t.end_time - t.start_time) if t.end_time else None
                }
                for t in self.task_history[-10:]
            ],
            "config": {
                "hourly_limit": self.config.hourly_limit_usd,
                "daily_limit": self.config.daily_limit_usd,
                "per_task_limit": self.config.per_task_limit_usd
            }
        }
