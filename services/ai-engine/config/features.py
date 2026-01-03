"""
AI Engine Feature Flags
Control which advanced features are enabled based on available dependencies
"""

import os
from typing import Dict, Any

class FeatureFlags:
    """
    Feature flags for AI Engine
    Automatically detects which features can be enabled
    """

    def __init__(self):
        # Check environment variables for explicit feature toggles
        self.ENABLE_KNOWLEDGE_GRAPH = self._check_feature('ENABLE_KNOWLEDGE_GRAPH', self._has_neo4j)
        self.ENABLE_ZERO_DAY_DISCOVERY = self._check_feature('ENABLE_ZERO_DAY_DISCOVERY', self._has_ml_deps)
        self.ENABLE_SIEM_GENERATION = self._check_feature('ENABLE_SIEM_GENERATION', self._has_basic_deps)
        self.ENABLE_THREAT_INTEL = self._check_feature('ENABLE_THREAT_INTEL', self._has_basic_deps)

    def _check_feature(self, env_var: str, dependency_check_func) -> bool:
        """
        Check if feature should be enabled
        Priority: Env var > Dependency check
        """
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value.lower() in ('true', '1', 'yes', 'on')
        return dependency_check_func()

    def _has_neo4j(self) -> bool:
        """Check if Neo4j dependencies are available"""
        try:
            from neo4j import GraphDatabase
            import networkx
            # Check if Neo4j server is reachable (optional)
            neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            # Don't actually connect, just check if library is available
            return True
        except ImportError:
            return False

    def _has_ml_deps(self) -> bool:
        """Check if ML dependencies are available"""
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False

    def _has_basic_deps(self) -> bool:
        """Check if basic dependencies are available"""
        try:
            import numpy
            import pandas
            return True
        except ImportError:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get status of all features"""
        return {
            'knowledge_graph': {
                'enabled': self.ENABLE_KNOWLEDGE_GRAPH,
                'description': 'Neo4j-based knowledge graph for attack-defense relationships',
                'dependencies': ['neo4j', 'networkx'],
                'requires_service': 'Neo4j server at NEO4J_URI'
            },
            'zero_day_discovery': {
                'enabled': self.ENABLE_ZERO_DAY_DISCOVERY,
                'description': 'ML-based zero-day vulnerability discovery',
                'dependencies': ['torch', 'transformers', 'scikit-learn'],
                'requires_service': None
            },
            'siem_rule_generation': {
                'enabled': self.ENABLE_SIEM_GENERATION,
                'description': 'Automated SIEM rule generation from attack patterns',
                'dependencies': ['numpy', 'pandas'],
                'requires_service': None
            },
            'threat_intelligence': {
                'enabled': self.ENABLE_THREAT_INTEL,
                'description': 'Threat intelligence aggregation and correlation',
                'dependencies': ['numpy', 'pandas'],
                'requires_service': None
            }
        }

# Global feature flags instance
features = FeatureFlags()

# Helper function to check if feature is enabled
def is_enabled(feature_name: str) -> bool:
    """Check if a specific feature is enabled"""
    return getattr(features, f'ENABLE_{feature_name.upper()}', False)
