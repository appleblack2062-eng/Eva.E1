"""Self-Reflection & Case Library for explanation-driven learning.

This module generates human-readable explanations of agent decisions and stores
cases for future case-based reasoning.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class Case:
    """A stored case with explanation and outcome."""
    pattern: str
    decision: str
    explanation: str
    outcome: Dict[str, Any]
    embedding: np.ndarray
    timestamp: float
    success: bool = True


class ReflectionEngine:
    """Generate explanations and store cases for future reasoning."""
    
    def __init__(self, llm_client=None, case_store=None):
        self.llm = llm_client
        self.case_store = case_store  # Vector store of past cases
        self._cases: List[Case] = []
        
        # Simple embedding function (can be replaced with actual model)
        self._embedding_dim = 384
    
    async def generate_explanation(
        self, 
        decision_context: Dict[str, Any], 
        outcome: Dict[str, Any]
    ) -> str:
        """Produce human-readable explanation of agent decision."""
        if not self.llm:
            return self._generate_simple_explanation(decision_context, outcome)
        
        prompt = f"""
Explain this agent decision clearly and concisely:
Decision: {decision_context.get('action', 'unknown')}
Reasoning: {decision_context.get('factors', {})}
Outcome: {outcome.get('result', 'unknown')} ({'success' if outcome.get('success', False) else 'failed'})

Format: 2-3 sentences, avoid jargon.
"""
        try:
            response = await self.llm.generate(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception:
            return self._generate_simple_explanation(decision_context, outcome)
    
    def _generate_simple_explanation(
        self, 
        decision_context: Dict[str, Any], 
        outcome: Dict[str, Any]
    ) -> str:
        """Generate simple explanation without LLM."""
        action = decision_context.get('action', 'unknown')
        success = outcome.get('success', False)
        
        if success:
            return f"The agent chose to {action} based on the current context, which resulted in successful task completion."
        else:
            return f"The agent attempted to {action}, but the task did not complete successfully. Review the factors for more details."
    
    async def store_case(
        self, 
        task_pattern: str, 
        decision: str, 
        explanation: str, 
        outcome: Dict[str, Any]
    ):
        """Store case for future retrieval."""
        embedding = self._encode_case(task_pattern, decision, outcome)
        
        case = Case(
            pattern=task_pattern,
            decision=decision,
            explanation=explanation,
            outcome=outcome,
            embedding=embedding,
            timestamp=time.time(),
            success=outcome.get('success', False)
        )
        
        self._cases.append(case)
        
        if self.case_store:
            await self.case_store.add({
                "pattern": task_pattern,
                "decision": decision,
                "explanation": explanation,
                "outcome": outcome,
                "embedding": embedding,
                "timestamp": time.time()
            })
    
    async def retrieve_similar_cases(
        self, 
        current_pattern: str, 
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Find past cases with similar patterns for case-based reasoning."""
        query_emb = self._encode_pattern(current_pattern)
        
        if self.case_store:
            return await self.case_store.search(query_emb, limit=limit)
        
        # Fallback to local case store
        return self._search_local_cases(query_emb, limit)
    
    def _search_local_cases(
        self, 
        query_emb: np.ndarray, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search local case list by similarity."""
        similarities = []
        
        for case in self._cases:
            sim = self._cosine_similarity(query_emb, case.embedding)
            similarities.append((case, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches as dicts
        results = []
        for case, sim in similarities[:limit]:
            results.append({
                "pattern": case.pattern,
                "decision": case.decision,
                "explanation": case.explanation,
                "outcome": case.outcome,
                "similarity": sim,
                "success": case.success
            })
        
        return results
    
    async def reason_from_cases(
        self, 
        current_task: Dict[str, Any], 
        similar_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use retrieved cases to inform current decision."""
        if not similar_cases:
            return {}
        
        # Extract lessons from cases
        lessons = [
            case["explanation"] 
            for case in similar_cases 
            if case.get("success", False) or case.get("outcome", {}).get("success", False)
        ]
        pitfalls = [
            case["explanation"] 
            for case in similar_cases 
            if not case.get("success", True) or not case.get("outcome", {}).get("success", True)
        ]
        
        if not self.llm:
            return self._simple_case_reasoning(current_task, lessons, pitfalls)
        
        # Synthesize guidance using LLM
        prompt = f"""
Based on these past experiences, advise on current task:
Task: {current_task}
Successful patterns: {lessons}
Failed patterns: {pitfalls}

Return JSON with: recommended_action, confidence, rationale.
"""
        try:
            response = await self.llm.generate(prompt, response_format="json")
            return response if isinstance(response, dict) else {}
        except Exception:
            return self._simple_case_reasoning(current_task, lessons, pitfalls)
    
    def _simple_case_reasoning(
        self, 
        current_task: Dict[str, Any], 
        lessons: List[str],
        pitfalls: List[str]
    ) -> Dict[str, Any]:
        """Simple case-based reasoning without LLM."""
        if lessons:
            return {
                "recommended_action": "follow_successful_pattern",
                "confidence": 0.7,
                "rationale": f"Previous successes suggest: {lessons[0]}"
            }
        elif pitfalls:
            return {
                "recommended_action": "avoid_failed_pattern",
                "confidence": 0.6,
                "rationale": f"Previous failures warn against: {pitfalls[0]}"
            }
        else:
            return {
                "recommended_action": "use_default_strategy",
                "confidence": 0.5,
                "rationale": "No relevant past cases found"
            }
    
    def _encode_case(
        self, 
        pattern: str, 
        decision: str, 
        outcome: Dict[str, Any]
    ) -> np.ndarray:
        """Encode case into embedding vector."""
        # Simple hash-based encoding (replace with actual model in production)
        text = f"{pattern}|{decision}|{outcome}"
        return self._text_to_embedding(text)
    
    def _encode_pattern(self, pattern: str) -> np.ndarray:
        """Encode task pattern into embedding vector."""
        return self._text_to_embedding(pattern)
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding using simple hash."""
        # Deterministic pseudo-embedding based on character hashes
        embedding = np.zeros(self._embedding_dim)
        for i, char in enumerate(text[:1000]):  # Limit text length
            idx = ord(char) % self._embedding_dim
            embedding[idx] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    async def replay_episodes(self, num_samples: int = 10):
        """Simulate 'what-if' scenarios on stored cases for learning."""
        if len(self._cases) < num_samples:
            return []
        
        # Sample random cases
        import random
        sampled_cases = random.sample(self._cases, min(num_samples, len(self._cases)))
        
        episodes = []
        for case in sampled_cases:
            # Simulate alternative decision
            episode = {
                "original_decision": case.decision,
                "original_outcome": case.outcome,
                "alternative_considered": self._generate_alternative(case),
            }
            episodes.append(episode)
        
        return episodes
    
    def _generate_alternative(self, case: Case) -> str:
        """Generate alternative decision for episodic replay."""
        # Simple heuristic: suggest opposite or different approach
        alternatives = {
            "optimize": "wait",
            "wait": "optimize",
            "deprecate": "retrain",
            "retrain": "deprecate",
        }
        return alternatives.get(case.decision, "use_different_strategy")
