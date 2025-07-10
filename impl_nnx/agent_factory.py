from typing import Dict, Any
from .base_agent import ContinualLearningAgent
from .vanilla_agent import VanillaSGDAgent
from .cbp_agent import CBPAgent
from .shrink_perturb_agent import ShrinkPerturbAgent

class AgentFactory:
    """Factory for creating continual learning agents"""
    
    _agents = {
        'vanilla': VanillaSGDAgent,
        'cbp': CBPAgent, 
        'shrink_perturb': ShrinkPerturbAgent,
    }
    
    @classmethod
    def create_agent(cls, agent_type: str, config: Dict[str, Any]) -> ContinualLearningAgent:
        if agent_type not in cls._agents:
            raise ValueError(f"Unknown agent type: {agent_type}. "
                           f"Available: {list(cls._agents.keys())}")
        
        return cls._agents[agent_type](config)
    
    @classmethod
    def register_agent(cls, name: str, agent_class: type):
        """Register a new agent type"""
        cls._agents[name] = agent_class
    
    @classmethod
    def list_agents(cls):
        return list(cls._agents.keys())