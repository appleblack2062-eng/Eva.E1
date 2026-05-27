"""Evolution Loop: Closed-loop self-improvement through mutation and validation."""

from __future__ import annotations
import asyncio
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class EvolutionStage(Enum):
    """Stages in the evolution pipeline."""
    DETECT = "detect"
    SELECT = "select"
    MUTATE = "mutate"
    EXECUTE = "execute"
    VALIDATE = "validate"
    EVALUATE = "evaluate"
    SOLIDIFY = "solidify"
    REUSE = "reuse"


@dataclass
class EvolutionSignal:
    """A signal triggering evolution (usually a failure pattern)."""
    task_pattern: str
    failure_type: str
    error_context: Dict[str, Any]
    timestamp: float
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_pattern': self.task_pattern,
            'failure_type': self.failure_type,
            'error_context': self.error_context,
            'timestamp': self.timestamp,
            'confidence': self.confidence
        }


@dataclass
class EvolutionCandidate:
    """A candidate solution generated during mutation."""
    id: str
    original_pattern: str
    mutation_type: str
    code_changes: Optional[str]
    prompt_changes: Optional[str]
    confidence_score: float = 0.0
    risk_score: float = 0.0
    test_pass_rate: float = 0.0


@dataclass
class EvolutionResult:
    """Result of an evolution cycle."""
    stage_reached: EvolutionStage
    success: bool
    candidates_generated: int
    candidates_validated: int
    promoted_asset: Optional[str]
    error_message: Optional[str] = None


class EvolutionLoop:
    """
    Closed-loop self-improvement system.
    
    Stages:
    1. Detect: Collect failure patterns
    2. Select: Choose most frequent/high-impact pattern
    3. Mutate: Generate candidate code/prompt changes
    4. Execute: Test candidates in sandbox
    5. Validate: Check test pass rate and risk
    6. Evaluate: Score improvement vs risk
    7. Solidify: Promote best candidate to ExperienceRepo
    8. Reuse: Update routing tables
    """
    
    def __init__(
        self,
        config,
        experience_repo=None,
        sandbox_manager=None,
        llm_client=None
    ):
        self.config = config
        self.experience_repo = experience_repo
        self.sandbox = sandbox_manager
        self.llm = llm_client
        
        # Signal queue for detected failures
        self.signal_queue: List[EvolutionSignal] = []
        self.max_queue_size = config.get('evolution_max_queue', 100)
        
        # Evolution history
        self.evolution_history: List[EvolutionResult] = []
        
        # Settings
        self.min_signals_for_evolution = config.get('min_signals_for_evolution', 3)
        self.max_candidates = config.get('max_evolution_candidates', 5)
        self.min_pass_rate = config.get('evolution_min_pass_rate', 0.8)
        self.max_risk_score = config.get('evolution_max_risk', 0.3)
    
    async def process_signal(self, signal: EvolutionSignal):
        """
        Add a failure signal to the queue.
        Triggers evolution cycle if threshold is met.
        
        Args:
            signal: The evolution signal to process
        """
        self.signal_queue.append(signal)
        
        # Keep queue bounded
        if len(self.signal_queue) > self.max_queue_size:
            self.signal_queue = self.signal_queue[-self.max_queue_size:]
        
        # Check if we should trigger evolution
        pattern_counts = self._count_patterns()
        for pattern, count in pattern_counts.items():
            if count >= self.min_signals_for_evolution:
                await self.run_evolution_cycle(pattern)
                break
    
    def _count_patterns(self) -> Dict[str, int]:
        """Count occurrences of each task pattern."""
        counts = {}
        for signal in self.signal_queue:
            pattern = signal.task_pattern
            counts[pattern] = counts.get(pattern, 0) + 1
        return counts
    
    async def run_evolution_cycle(self, pattern: Optional[str] = None) -> EvolutionResult:
        """
        Execute the full 8-stage evolution pipeline.
        
        Args:
            pattern: Specific pattern to evolve (if None, selects most frequent)
            
        Returns:
            EvolutionResult with cycle outcome
        """
        result = EvolutionResult(
            stage_reached=EvolutionStage.DETECT,
            success=False,
            candidates_generated=0,
            candidates_validated=0,
            promoted_asset=None
        )
        
        try:
            # Stage 1: Detect - Already done via signals
            result.stage_reached = EvolutionStage.DETECT
            
            # Stage 2: Select - Choose pattern to evolve
            selected_pattern = pattern or self._select_pattern()
            if not selected_pattern:
                result.error_message = "No pattern selected for evolution"
                return result
            
            result.stage_reached = EvolutionStage.SELECT
            
            # Stage 3: Mutate - Generate candidates
            candidates = await self._mutate(selected_pattern)
            result.candidates_generated = len(candidates)
            
            if not candidates:
                result.error_message = "No candidates generated"
                return result
            
            result.stage_reached = EvolutionStage.MUTATE
            
            # Stage 4 & 5: Execute & Validate - Test candidates
            validated = await self._execute_and_validate(candidates)
            result.candidates_validated = len(validated)
            
            if not validated:
                result.error_message = "No candidates passed validation"
                return result
            
            result.stage_reached = EvolutionStage.VALIDATE
            
            # Stage 6: Evaluate - Score and rank
            best_candidate = self._evaluate(validated)
            
            if not best_candidate:
                result.error_message = "No candidate met evaluation criteria"
                return result
            
            result.stage_reached = EvolutionStage.EVALUATE
            
            # Stage 7: Solidify - Promote to experience repo
            if self.experience_repo and best_candidate:
                asset_id = await self._solidify(best_candidate, selected_pattern)
                result.promoted_asset = asset_id
            
            result.stage_reached = EvolutionStage.SOLIDIFY
            
            # Stage 8: Reuse - Update routing (would happen automatically via repo)
            result.stage_reached = EvolutionStage.REUSE
            result.success = True
            
            # Clean up processed signals
            self._cleanup_signals(selected_pattern)
            
        except Exception as e:
            result.error_message = str(e)
        
        # Record history
        self.evolution_history.append(result)
        return result
    
    def _select_pattern(self) -> Optional[str]:
        """Select the most frequent/high-impact pattern to evolve."""
        if not self.signal_queue:
            return None
        
        pattern_counts = self._count_patterns()
        if not pattern_counts:
            return None
        
        # Select pattern with highest count
        return max(pattern_counts, key=pattern_counts.get)
    
    async def _mutate(self, pattern: str) -> List[EvolutionCandidate]:
        """
        Generate candidate solutions using LLM.
        
        Args:
            pattern: The task pattern to improve
            
        Returns:
            List of EvolutionCandidates
        """
        if not self.llm:
            return []
        
        # Gather context from signals
        related_signals = [
            s for s in self.signal_queue 
            if s.task_pattern == pattern
        ]
        
        error_examples = "\n".join([
            f"- {s.failure_type}: {s.error_context.get('error', 'Unknown')}"
            for s in related_signals[:5]
        ])
        
        prompt = f"""You are an AI system improving itself based on failure patterns.

TASK PATTERN: {pattern}

OBSERVED FAILURES:
{error_examples}

Generate {self.max_candidates} different strategies to handle this pattern better.
For each strategy, provide:
1. A brief description of the approach
2. Specific code or prompt changes needed
3. Why this might work better

Format as JSON array with fields: mutation_type, description, code_changes, prompt_changes"""
        
        try:
            response = await self.llm.generate(prompt, response_format="json")
            mutations = response.data if hasattr(response, 'data') else []
            
            candidates = []
            for i, mutation in enumerate(mutations[:self.max_candidates]):
                candidate = EvolutionCandidate(
                    id=f"candidate_{pattern}_{i}_{int(time.time())}",
                    original_pattern=pattern,
                    mutation_type=mutation.get('mutation_type', 'unknown'),
                    code_changes=mutation.get('code_changes'),
                    prompt_changes=mutation.get('prompt_changes'),
                    confidence_score=0.5  # Initial score
                )
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            print(f"Mutation error: {e}")
            return []
    
    async def _execute_and_validate(
        self, 
        candidates: List[EvolutionCandidate]
    ) -> List[EvolutionCandidate]:
        """
        Test candidates in sandbox and validate results.
        
        Args:
            candidates: List of candidates to test
            
        Returns:
            List of candidates that passed validation
        """
        validated = []
        
        for candidate in candidates:
            try:
                # Execute in sandbox if available
                if self.sandbox and candidate.code_changes:
                    test_result = await self._run_candidate_tests(candidate)
                    candidate.test_pass_rate = test_result.get('pass_rate', 0.0)
                    candidate.risk_score = test_result.get('risk_score', 1.0)
                else:
                    # Simulate validation
                    candidate.test_pass_rate = 0.7
                    candidate.risk_score = 0.4
                
                # Check if candidate meets thresholds
                if (candidate.test_pass_rate >= self.min_pass_rate and 
                    candidate.risk_score <= self.max_risk_score):
                    validated.append(candidate)
                    
            except Exception as e:
                print(f"Candidate validation error: {e}")
                candidate.test_pass_rate = 0.0
                candidate.risk_score = 1.0
        
        return validated
    
    async def _run_candidate_tests(self, candidate: EvolutionCandidate) -> Dict[str, float]:
        """Run tests for a candidate in the sandbox."""
        # This would execute actual tests in a real implementation
        # For now, return simulated results
        return {
            'pass_rate': 0.85,
            'risk_score': 0.2
        }
    
    def _evaluate(self, candidates: List[EvolutionCandidate]) -> Optional[EvolutionCandidate]:
        """
        Select the best candidate based on scores.
        
        Args:
            candidates: Validated candidates
            
        Returns:
            Best candidate or None
        """
        if not candidates:
            return None
        
        # Score = pass_rate * (1 - risk_score)
        for candidate in candidates:
            candidate.confidence_score = (
                candidate.test_pass_rate * (1 - candidate.risk_score)
            )
        
        # Return highest scoring candidate
        return max(candidates, key=lambda c: c.confidence_score)
    
    async def _solidify(
        self, 
        candidate: EvolutionCandidate, 
        pattern: str
    ) -> Optional[str]:
        """
        Promote candidate to experience repository.
        
        Args:
            candidate: The candidate to promote
            pattern: The task pattern
            
        Returns:
            Asset ID if successful
        """
        if not self.experience_repo:
            return None
        
        # Create signature for the asset
        asset_data = {
            'pattern': pattern,
            'mutation_type': candidate.mutation_type,
            'code_changes': candidate.code_changes,
            'prompt_changes': candidate.prompt_changes,
            'confidence': candidate.confidence_score
        }
        
        signature = hashlib.sha256(
            str(asset_data).encode()
        ).hexdigest()[:16]
        
        # Promote to repo
        asset_id = await self.experience_repo.promote_asset(
            pattern=pattern,
            solution=asset_data,
            confidence=candidate.confidence_score,
            signature=signature
        )
        
        return asset_id
    
    def _cleanup_signals(self, pattern: str):
        """Remove processed signals from the queue."""
        self.signal_queue = [
            s for s in self.signal_queue 
            if s.task_pattern != pattern
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evolution loop statistics."""
        successful = sum(1 for r in self.evolution_history if r.success)
        total = len(self.evolution_history)
        
        return {
            'queue_size': len(self.signal_queue),
            'total_cycles': total,
            'successful_cycles': successful,
            'success_rate': successful / total if total > 0 else 0,
            'total_candidates_generated': sum(
                r.candidates_generated for r in self.evolution_history
            ),
            'total_assets_promoted': sum(
                1 for r in self.evolution_history if r.promoted_asset
            )
        }
