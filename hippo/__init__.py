"""
Hippo: Dual-AI Cascade Memory System

A neuromorphic dual-agent AI system inspired by the hippocampus-neocortex
memory architecture in human cognition.
"""

from hippo.volatile_agent import VolatileAgent
from hippo.neocortex_agent import NeocortexAgent
from hippo.semantic_bridge import SemanticBridge
from hippo.loop import run_dual_cascade, InferenceCycle

__version__ = "0.1.0"

__all__ = [
    "VolatileAgent",
    "NeocortexAgent",
    "SemanticBridge",
    "run_dual_cascade",
    "InferenceCycle",
]
