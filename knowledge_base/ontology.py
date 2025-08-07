"""
Regulatory Ontology Management

Handles the creation and management of ontologies for regulatory compliance.
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from owlready2 import get_ontology, Thing, DataProperty, ObjectProperty, FunctionalProperty
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL

logger = logging.getLogger(__name__)


class RegulatoryOntology(ABC):
    """Abstract base class for regulatory ontologies."""
    
    def __init__(self, ontology_uri: str):
        """
        Initialize the regulatory ontology.
        
        Args:
            ontology_uri: URI for the ontology namespace
        """
        self.ontology_uri = ontology_uri
        self.ontology = get_ontology(ontology_uri)
        self.namespace = Namespace(ontology_uri)
        
    @abstractmethod
    def define_classes(self) -> None:
        """Define the main classes for this regulatory domain."""
        pass
        
    @abstractmethod  
    def define_properties(self) -> None:
        """Define object and data properties for this regulatory domain."""
        pass
        
    @abstractmethod
    def define_rules(self) -> None:
        """Define logical rules and constraints for this regulatory domain."""
        pass
        
    def save(self, output_path: Path) -> None:
        """Save the ontology to a file."""
        self.ontology.save(file=str(output_path))
        
    def load(self, ontology_path: Path) -> None:
        """Load ontology from a file."""
        self.ontology.load(file=str(ontology_path))


class EUAIActOntology(RegulatoryOntology):
    """
    Ontology specifically for the EU AI Act regulations.
    
    Defines classes, properties, and rules specific to EU AI Act compliance.
    """
    
    def __init__(self):
        super().__init__("http://ai-rosetta-stone.org/eu-ai-act#")
        self.classes = {}
        self.properties = {}
        self.individuals = {}
        
        # Initialize the ontology structure
        self.define_classes()
        self.define_properties() 
        self.define_rules()
        
    def define_classes(self) -> None:
        """Define the main classes for EU AI Act compliance."""
        
        with self.ontology:
            # Core AI System Classes
            class AISystem(Thing):
                """An AI system as defined by the EU AI Act."""
                pass
                
            class HighRiskAISystem(AISystem):
                """AI systems classified as high-risk under Annex III."""
                pass
                
            class LimitedRiskAISystem(AISystem):
                """AI systems with limited risk requiring transparency obligations."""
                pass
                
            class MinimalRiskAISystem(AISystem):
                """AI systems with minimal risk and limited obligations."""
                pass
                
            class ProhibitedAISystem(AISystem):
                """AI systems that are prohibited under Article 5."""
                pass
                
            # Regulatory Requirement Classes
            class RegulatoryRequirement(Thing):
                """A requirement imposed by the EU AI Act."""
                pass
                
            class TransparencyRequirement(RegulatoryRequirement):
                """Transparency and information requirements."""
                pass
                
            class HumanOversightRequirement(RegulatoryRequirement):
                """Human oversight requirements."""
                pass
                
            class DataGovernanceRequirement(RegulatoryRequirement):
                """Data and data governance requirements."""
                pass
                
            class RobustnessRequirement(RegulatoryRequirement):
                """Accuracy, robustness and cybersecurity requirements."""
                pass
                
            # Compliance Classes
            class ComplianceStatus(Thing):
                """Compliance status of an AI system."""
                pass
                
            class Article(Thing):
                """An article from the EU AI Act."""
                pass
                
            # Store classes for later reference
            self.classes.update({
                'AISystem': AISystem,
                'HighRiskAISystem': HighRiskAISystem,
                'LimitedRiskAISystem': LimitedRiskAISystem,
                'MinimalRiskAISystem': MinimalRiskAISystem,
                'ProhibitedAISystem': ProhibitedAISystem,
                'RegulatoryRequirement': RegulatoryRequirement,
                'TransparencyRequirement': TransparencyRequirement,
                'HumanOversightRequirement': HumanOversightRequirement,
                'DataGovernanceRequirement': DataGovernanceRequirement,
                'RobustnessRequirement': RobustnessRequirement,
                'ComplianceStatus': ComplianceStatus,
                'Article': Article
            })
            
    def define_properties(self) -> None:
        """Define object and data properties for EU AI Act compliance."""
        
        with self.ontology:
            # Object Properties
            class hasRequirement(ObjectProperty):
                """Links an AI system to its regulatory requirements."""
                domain = [self.classes['AISystem']]
                range = [self.classes['RegulatoryRequirement']]
                
            class isSubjectTo(ObjectProperty):
                """Links an AI system to applicable articles."""
                domain = [self.classes['AISystem']]
                range = [self.classes['Article']]
                
            class hasComplianceStatus(ObjectProperty, FunctionalProperty):
                """Links an AI system to its compliance status."""
                domain = [self.classes['AISystem']]
                range = [self.classes['ComplianceStatus']]
                
            class derivedFrom(ObjectProperty):
                """Links a requirement to its source article."""
                domain = [self.classes['RegulatoryRequirement']]
                range = [self.classes['Article']]
                
            # Data Properties
            class hasRiskLevel(DataProperty, FunctionalProperty):
                """Risk level of an AI system."""
                domain = [self.classes['AISystem']]
                range = [str]
                
            class hasArticleNumber(DataProperty, FunctionalProperty):
                """Article number in the EU AI Act."""
                domain = [self.classes['Article']]
                range = [str]
                
            class hasTitle(DataProperty, FunctionalProperty):
                """Title of an article or requirement."""
                range = [str]
                
            class hasDescription(DataProperty, FunctionalProperty):
                """Description of an article or requirement."""
                range = [str]
                
            class isCompliant(DataProperty, FunctionalProperty):
                """Boolean indicating compliance status."""
                domain = [self.classes['AISystem']]
                range = [bool]
                
            class hasConfidenceScore(DataProperty, FunctionalProperty):
                """Confidence score for compliance assessment."""
                domain = [self.classes['ComplianceStatus']]
                range = [float]
                
            # Store properties for later reference
            self.properties.update({
                'hasRequirement': hasRequirement,
                'isSubjectTo': isSubjectTo,
                'hasComplianceStatus': hasComplianceStatus,
                'derivedFrom': derivedFrom,
                'hasRiskLevel': hasRiskLevel,
                'hasArticleNumber': hasArticleNumber,
                'hasTitle': hasTitle,
                'hasDescription': hasDescription,
                'isCompliant': isCompliant,
                'hasConfidenceScore': hasConfidenceScore
            })
            
    def define_rules(self) -> None:
        """Define logical rules and constraints for EU AI Act compliance."""
        
        # TODO: Implement SWRL rules or similar for logical constraints
        # Examples:
        # - High-risk AI systems must have human oversight
        # - Systems using biometric data are subject to additional requirements
        # - Prohibited practices are never compliant
        
        logger.info("EU AI Act ontology rules defined")
        
    def create_article_individual(self, article_number: str, title: str, description: str) -> Any:
        """
        Create an individual article in the ontology.
        
        Args:
            article_number: Article number (e.g., "10", "13", "14")
            title: Article title
            description: Article description
            
        Returns:
            The created article individual
        """
        with self.ontology:
            article_name = f"Article_{article_number}"
            article = self.classes['Article'](article_name)
            article.hasArticleNumber = [article_number]
            article.hasTitle = [title]
            article.hasDescription = [description]
            
            self.individuals[article_name] = article
            return article
            
    def create_ai_system_individual(self, system_name: str, risk_level: str) -> Any:
        """
        Create an AI system individual in the ontology.
        
        Args:
            system_name: Name/identifier of the AI system
            risk_level: Risk level ('high', 'limited', 'minimal', 'prohibited')
            
        Returns:
            The created AI system individual
        """
        with self.ontology:
            # Choose appropriate class based on risk level
            system_class = {
                'high': self.classes['HighRiskAISystem'],
                'limited': self.classes['LimitedRiskAISystem'], 
                'minimal': self.classes['MinimalRiskAISystem'],
                'prohibited': self.classes['ProhibitedAISystem']
            }.get(risk_level.lower(), self.classes['AISystem'])
            
            system = system_class(system_name)
            system.hasRiskLevel = [risk_level]
            
            self.individuals[system_name] = system
            return system
            
    def get_applicable_requirements(self, system_name: str) -> List[str]:
        """
        Get requirements applicable to a specific AI system.
        
        Args:
            system_name: Name of the AI system
            
        Returns:
            List of applicable requirement names
        """
        # TODO: Implement SPARQL query to find applicable requirements
        # based on system classification and risk level
        pass
        
    def validate_ontology(self) -> Tuple[bool, List[str]]:
        """
        Validate the ontology for consistency.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        # TODO: Implement ontology validation using reasoner
        pass