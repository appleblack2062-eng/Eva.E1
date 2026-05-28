"""Implicit Feedback Learner for learning from user edits.

This module extracts learning signals from user modifications without requiring explicit labels.
"""

from __future__ import annotations
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class FeedbackSignal:
    """Signal extracted from user feedback."""
    task_context: Dict[str, Any]
    original: str
    edited: str
    diff: Dict[str, Any]  # {added, removed, modified}
    change_type: str  # "format", "value", "structure"
    preference: Dict[str, Optional[str]]  # {avoid, prefer}
    confidence: float


class ImplicitFeedbackParser:
    """Extract learning signals from user modifications."""
    
    def __init__(self, diff_model_name: str = None):
        # Simple diff-based approach (can be enhanced with codebert)
        self.diff_model_name = diff_model_name
    
    def parse_user_edit(
        self, 
        original_output: str, 
        user_edited: str, 
        context: Dict[str, Any]
    ) -> FeedbackSignal:
        """Analyze what the user changed and why."""
        # Compute diff
        diff = self._compute_diff(original_output, user_edited)
        
        # Classify change type
        change_type = self._classify_change(diff, context)
        
        # Infer preference
        preference = {
            "avoid": change_type if self._is_correction(diff) else None,
            "prefer": self._extract_positive_pattern(diff) if self._is_improvement(diff) else None,
        }
        
        return FeedbackSignal(
            task_context=context,
            original=original_output,
            edited=user_edited,
            diff=diff,
            change_type=change_type,
            preference=preference,
            confidence=self._estimate_confidence(diff)
        )
    
    def integrate_feedback(
        self, 
        signal: FeedbackSignal, 
        workflow
    ):
        """Use feedback to refine workflow."""
        if signal.preference.get("avoid"):
            # Find step producing the problematic output pattern
            problematic_step = self._locate_responsible_step(workflow, signal.diff)
            
            if problematic_step:
                # Generate fix prompt
                prompt = f"""
Avoid this pattern in output: {signal.diff.get('removed', '')}
Step context: {problematic_step}
Suggest modification to prevent this issue.
"""
                # Apply LLM-generated fix (simplified)
                return self._apply_workflow_fix(workflow, problematic_step, prompt)
        
        if signal.preference.get("prefer"):
            # Reinforce the preferred pattern
            return self._reinforce_pattern(workflow, signal.preference["prefer"])
        
        return workflow
    
    def _compute_diff(self, original: str, edited: str) -> Dict[str, Any]:
        """Compute simple diff between two strings."""
        original_lines = set(original.split('\n'))
        edited_lines = set(edited.split('\n'))
        
        removed = original_lines - edited_lines
        added = edited_lines - original_lines
        
        return {
            "added": list(added),
            "removed": list(removed),
            "unchanged": list(original_lines & edited_lines),
        }
    
    def _classify_change(self, diff: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Classify the type of change made by user."""
        added_text = ' '.join(diff.get("added", []))
        removed_text = ' '.join(diff.get("removed", []))
        
        # Check for format changes (whitespace, punctuation)
        if self._is_format_change(added_text, removed_text):
            return "format"
        
        # Check for value changes (numbers, strings)
        if self._is_value_change(added_text, removed_text):
            return "value"
        
        # Check for structure changes (ordering, nesting)
        if self._is_structure_change(added_text, removed_text):
            return "structure"
        
        return "other"
    
    def _is_format_change(self, added: str, removed: str) -> bool:
        """Check if change is primarily formatting."""
        # Normalize whitespace and compare
        added_norm = re.sub(r'\s+', ' ', added.strip())
        removed_norm = re.sub(r'\s+', ' ', removed.strip())
        return added_norm == removed_norm
    
    def _is_value_change(self, added: str, removed: str) -> bool:
        """Check if change involves value modifications."""
        # Look for numbers, quoted strings, etc.
        number_pattern = r'\d+\.?\d*'
        has_number_change = bool(re.search(number_pattern, added)) != bool(re.search(number_pattern, removed))
        
        # Look for string literal changes
        string_pattern = r'["\'][^"\']*["\']'
        has_string_change = bool(re.search(string_pattern, added)) != bool(re.search(string_pattern, removed))
        
        return has_number_change or has_string_change
    
    def _is_structure_change(self, added: str, removed: str) -> bool:
        """Check if change involves structural modifications."""
        # Look for bracket/brace changes
        brackets = ['{', '}', '[', ']', '(', ')']
        added_brackets = sum(added.count(b) for b in brackets)
        removed_brackets = sum(removed.count(b) for b in brackets)
        
        return added_brackets != removed_brackets
    
    def _is_correction(self, diff: Dict[str, Any]) -> bool:
        """Determine if the change appears to be a correction."""
        # Heuristic: if content was removed and replaced, likely a correction
        return len(diff.get("removed", [])) > 0 and len(diff.get("added", [])) > 0
    
    def _is_improvement(self, diff: Dict[str, Any]) -> bool:
        """Determine if the change appears to be an improvement."""
        # Heuristic: if only additions (no removals), likely an enhancement
        return len(diff.get("added", [])) > 0 and len(diff.get("removed", [])) == 0
    
    def _extract_positive_pattern(self, diff: Dict[str, Any]) -> Optional[str]:
        """Extract pattern that should be preferred."""
        added = diff.get("added", [])
        if added:
            return ' '.join(added[:3])  # Return first few additions
        return None
    
    def _estimate_confidence(self, diff: Dict[str, Any]) -> float:
        """Estimate confidence in the inferred preference."""
        # More changes = higher confidence
        total_changes = len(diff.get("added", [])) + len(diff.get("removed", []))
        
        if total_changes == 0:
            return 0.0
        elif total_changes < 3:
            return 0.5
        elif total_changes < 10:
            return 0.7
        else:
            return 0.9
    
    def _locate_responsible_step(self, workflow, diff: Dict[str, Any]) -> Optional[Any]:
        """Find the workflow step responsible for the problematic output."""
        if not hasattr(workflow, 'steps'):
            return None
        
        removed_content = ' '.join(diff.get("removed", []))
        
        # Search for step that might have produced this content
        for step in workflow.steps:
            step_desc = str(step)
            if any(word in step_desc for word in removed_content.split()[:5]):
                return step
        
        # Default: return last step
        if workflow.steps:
            return workflow.steps[-1]
        
        return None
    
    def _apply_workflow_fix(self, workflow, step_id: Any, prompt: str):
        """Apply fix to workflow based on feedback."""
        # Simplified: just mark the step for review
        if hasattr(workflow, 'steps'):
            for i, step in enumerate(workflow.steps):
                current_id = step.get('id', i) if isinstance(step, dict) else getattr(step, 'id', i)
                if current_id == step_id:
                    if isinstance(step, dict):
                        step['needs_review'] = True
                        step['feedback_prompt'] = prompt
                    else:
                        setattr(step, 'needs_review', True)
                        setattr(step, 'feedback_prompt', prompt)
                    break
        return workflow
    
    def _reinforce_pattern(self, workflow, pattern: str):
        """Reinforce a preferred pattern in the workflow."""
        # Add pattern to workflow metadata for future reference
        if hasattr(workflow, 'metadata'):
            if isinstance(workflow.metadata, dict):
                preferred_patterns = workflow.metadata.get('preferred_patterns', [])
                preferred_patterns.append(pattern)
                workflow.metadata['preferred_patterns'] = preferred_patterns
            else:
                setattr(workflow, 'preferred_patterns', [pattern])
        return workflow
