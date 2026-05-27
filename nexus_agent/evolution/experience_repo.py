"""Experience Repository: Store and retrieve proven solutions with cryptographic signatures."""

from __future__ import annotations
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ExperienceAsset:
    """A proven solution asset."""
    id: str
    pattern: str
    solution: Dict[str, Any]
    confidence: float
    signature: str
    created_at: float
    usage_count: int = 0
    last_used: Optional[float] = None
    version: int = 1


class ExperienceRepo:
    """
    Repository for storing and retrieving proven solutions.
    Assets are cryptographically signed to ensure integrity.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory index for fast lookups
        self.assets: Dict[str, ExperienceAsset] = {}
        self.pattern_index: Dict[str, List[str]] = {}  # pattern -> [asset_ids]
        
        # Load existing assets
        self._load_assets()
        
        # Statistics
        self.total_retrievals = 0
        self.total_promotions = 0
    
    def _load_assets(self):
        """Load assets from disk."""
        index_file = self.storage_path / "assets_index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    
                    for asset_id, asset_data in data.items():
                        asset = ExperienceAsset(**asset_data)
                        self.assets[asset_id] = asset
                        
                        # Build pattern index
                        pattern = asset.pattern
                        if pattern not in self.pattern_index:
                            self.pattern_index[pattern] = []
                        self.pattern_index[pattern].append(asset_id)
                        
            except Exception as e:
                print(f"Error loading experience repo: {e}")
    
    def _save_assets(self):
        """Save assets to disk."""
        index_file = self.storage_path / "assets_index.json"
        
        try:
            data = {
                asset_id: asdict(asset)
                for asset_id, asset in self.assets.items()
            }
            
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving experience repo: {e}")
    
    async def promote_asset(
        self,
        pattern: str,
        solution: Dict[str, Any],
        confidence: float,
        signature: str
    ) -> str:
        """
        Promote a new solution to the repository.
        
        Args:
            pattern: Task pattern this solution addresses
            solution: The solution data (code changes, prompts, etc.)
            confidence: Confidence score (0-1)
            signature: Cryptographic signature for integrity
            
        Returns:
            Asset ID
        """
        # Generate unique ID
        asset_id = f"asset_{pattern}_{hashlib.sha256(str(solution).encode()).hexdigest()[:8]}"
        
        # Create asset
        asset = ExperienceAsset(
            id=asset_id,
            pattern=pattern,
            solution=solution,
            confidence=confidence,
            signature=signature,
            created_at=time.time(),
            version=1
        )
        
        # Check if we already have an asset for this pattern
        existing = self.get_asset(pattern)
        if existing:
            # Update version if new one is better
            if confidence > existing.confidence:
                asset.version = existing.version + 1
                # Mark old one as superseded (could archive it)
            else:
                # Don't promote if worse than existing
                return existing.id
        
        # Store asset
        self.assets[asset_id] = asset
        
        # Update pattern index
        if pattern not in self.pattern_index:
            self.pattern_index[pattern] = []
        self.pattern_index[pattern].append(asset_id)
        
        # Save to disk
        self._save_assets()
        
        self.total_promotions += 1
        return asset_id
    
    def get_asset(self, pattern: str) -> Optional[ExperienceAsset]:
        """
        Retrieve the best known solution for a pattern.
        
        Args:
            pattern: Task pattern to look up
            
        Returns:
            Best ExperienceAsset or None
        """
        if pattern not in self.pattern_index:
            return None
        
        asset_ids = self.pattern_index[pattern]
        
        # Get all assets for this pattern
        candidates = [
            self.assets[aid] for aid in asset_ids
            if aid in self.assets
        ]
        
        if not candidates:
            return None
        
        # Return highest confidence asset
        best = max(candidates, key=lambda a: a.confidence)
        
        # Update usage stats
        best.usage_count += 1
        best.last_used = time.time()
        self._save_assets()
        
        self.total_retrievals += 1
        return best
    
    def list_assets(
        self, 
        pattern_filter: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> List[ExperienceAsset]:
        """
        List assets with optional filtering.
        
        Args:
            pattern_filter: Only return assets matching this pattern
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            List of ExperienceAssets
        """
        results = []
        
        for asset in self.assets.values():
            # Apply filters
            if pattern_filter and asset.pattern != pattern_filter:
                continue
            
            if asset.confidence < min_confidence:
                continue
            
            results.append(asset)
        
        # Sort by confidence (highest first)
        results.sort(key=lambda a: a.confidence, reverse=True)
        
        return results[:limit]
    
    def delete_asset(self, asset_id: str) -> bool:
        """
        Remove an asset from the repository.
        
        Args:
            asset_id: ID of asset to delete
            
        Returns:
            True if deleted successfully
        """
        if asset_id not in self.assets:
            return False
        
        asset = self.assets[asset_id]
        
        # Remove from pattern index
        if asset.pattern in self.pattern_index:
            self.pattern_index[asset.pattern].remove(asset_id)
            if not self.pattern_index[asset.pattern]:
                del self.pattern_index[asset.pattern]
        
        # Remove from assets
        del self.assets[asset_id]
        
        # Save changes
        self._save_assets()
        
        return True
    
    def verify_signature(self, asset_id: str) -> bool:
        """
        Verify the cryptographic signature of an asset.
        
        Args:
            asset_id: ID of asset to verify
            
        Returns:
            True if signature is valid
        """
        if asset_id not in self.assets:
            return False
        
        asset = self.assets[asset_id]
        
        # Recompute signature
        asset_data = {
            'pattern': asset.pattern,
            'solution': asset.solution,
            'confidence': asset.confidence
        }
        
        expected_signature = hashlib.sha256(
            str(asset_data).encode()
        ).hexdigest()[:16]
        
        return asset.signature == expected_signature
    
    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        patterns_covered = len(self.pattern_index)
        avg_confidence = (
            sum(a.confidence for a in self.assets.values()) / len(self.assets)
            if self.assets else 0
        )
        
        return {
            'total_assets': len(self.assets),
            'patterns_covered': patterns_covered,
            'total_promotions': self.total_promotions,
            'total_retrievals': self.total_retrievals,
            'average_confidence': avg_confidence,
            'most_used_pattern': (
                max(self.pattern_index.keys(), 
                    key=lambda p: sum(self.assets[aid].usage_count for aid in self.pattern_index[p]))
                if self.pattern_index else None
            )
        }
    
    def export_assets(self, output_path: str) -> bool:
        """
        Export all assets to a JSON file.
        
        Args:
            output_path: Path to export file
            
        Returns:
            True if successful
        """
        try:
            data = {
                'exported_at': time.time(),
                'total_assets': len(self.assets),
                'assets': [asdict(a) for a in self.assets.values()]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def import_assets(self, input_path: str) -> int:
        """
        Import assets from a JSON file.
        
        Args:
            input_path: Path to import file
            
        Returns:
            Number of assets imported
        """
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            imported = 0
            for asset_data in data.get('assets', []):
                asset = ExperienceAsset(**asset_data)
                
                if asset.id not in self.assets:
                    self.assets[asset.id] = asset
                    
                    if asset.pattern not in self.pattern_index:
                        self.pattern_index[asset.pattern] = []
                    self.pattern_index[asset.pattern].append(asset.id)
                    
                    imported += 1
            
            self._save_assets()
            return imported
            
        except Exception as e:
            print(f"Import error: {e}")
            return 0
