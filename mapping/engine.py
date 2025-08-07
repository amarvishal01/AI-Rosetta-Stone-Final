"""
Mapping & Reasoning Engine

Main engine that compares extracted model rules against legal knowledge base
and performs automated compliance reasoning.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import logging
from dataclasses import dataclass
from enum import Enum

from ..knowledge_base.builder import SymbolicKnowledgeBase, SymbolicRule as LegalRule
from ..bridge.extractor import ExtractedRule
from ..bridge.converter import SymbolicRule as ModelRule

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Possible compliance status values."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNKNOWN = "unknown"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class ComplianceViolation:
    """Represents a compliance violation found during analysis."""
    violation_id: str
    rule_id: str
    legal_requirement: str
    violation_type: str
    severity: str  # 'critical', 'major', 'minor', 'warning'
    description: str
    suggested_remediation: str
    confidence: float


@dataclass
class ComplianceMapping:
    """Represents a mapping between model rule and legal requirement."""
    mapping_id: str
    model_rule: ModelRule
    legal_rule: LegalRule
    mapping_type: str  # 'direct', 'indirect', 'conflicting', 'supporting'
    confidence: float
    explanation: str


@dataclass
class ComplianceAssessment:
    """Complete compliance assessment result."""
    system_id: str
    overall_status: ComplianceStatus
    confidence_score: float
    mappings: List[ComplianceMapping]
    violations: List[ComplianceViolation]
    recommendations: List[str]
    assessment_summary: str


class ComplianceMapper:
    """
    Maps model rules to legal requirements and identifies relationships.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize the compliance mapper.
        
        Args:
            similarity_threshold: Minimum similarity for rule mapping
        """
        self.similarity_threshold = similarity_threshold
        self.mappings: List[ComplianceMapping] = []
        
    def map_rules_to_requirements(self, model_rules: List[ModelRule],
                                legal_rules: List[LegalRule]) -> List[ComplianceMapping]:
        """
        Map model rules to legal requirements.
        
        Args:
            model_rules: Rules extracted from the AI model
            legal_rules: Rules from the legal knowledge base
            
        Returns:
            List of compliance mappings
        """
        logger.info(f"Mapping {len(model_rules)} model rules to {len(legal_rules)} legal requirements")
        
        mappings = []
        
        for model_rule in model_rules:
            # Find relevant legal rules for this model rule
            relevant_legal_rules = self._find_relevant_legal_rules(model_rule, legal_rules)
            
            for legal_rule, similarity in relevant_legal_rules:
                if similarity >= self.similarity_threshold:
                    mapping = self._create_mapping(model_rule, legal_rule, similarity)
                    mappings.append(mapping)
                    
        self.mappings = mappings
        logger.info(f"Created {len(mappings)} rule mappings")
        return mappings
        
    def _find_relevant_legal_rules(self, model_rule: ModelRule, 
                                 legal_rules: List[LegalRule]) -> List[Tuple[LegalRule, float]]:
        """Find legal rules relevant to a model rule."""
        relevant_rules = []
        
        for legal_rule in legal_rules:
            similarity = self._calculate_rule_similarity(model_rule, legal_rule)
            if similarity > 0.1:  # Only include rules with some similarity
                relevant_rules.append((legal_rule, similarity))
                
        # Sort by similarity (highest first)
        relevant_rules.sort(key=lambda x: x[1], reverse=True)
        return relevant_rules
        
    def _calculate_rule_similarity(self, model_rule: ModelRule, legal_rule: LegalRule) -> float:
        """Calculate similarity between a model rule and legal rule."""
        # TODO: Implement sophisticated rule similarity calculation
        # This could include:
        # - Semantic similarity of rule text
        # - Overlap in variables/concepts
        # - Structural similarity of logical expressions
        # - Domain-specific similarity measures
        
        # Placeholder implementation using simple text similarity
        model_text = model_rule.natural_language.lower()
        legal_text = f"{legal_rule.premise} {legal_rule.conclusion}".lower()
        
        # Simple word overlap similarity
        model_words = set(model_text.split())
        legal_words = set(legal_text.split())
        
        if not model_words or not legal_words:
            return 0.0
            
        overlap = len(model_words.intersection(legal_words))
        union = len(model_words.union(legal_words))
        
        return overlap / union if union > 0 else 0.0
        
    def _create_mapping(self, model_rule: ModelRule, legal_rule: LegalRule, 
                       similarity: float) -> ComplianceMapping:
        """Create a compliance mapping between model and legal rules."""
        mapping_id = f"mapping_{len(self.mappings)}"
        
        # Determine mapping type based on similarity and rule content
        if similarity > 0.8:
            mapping_type = "direct"
        elif similarity > 0.6:
            mapping_type = "indirect"
        elif self._rules_conflict(model_rule, legal_rule):
            mapping_type = "conflicting"
        else:
            mapping_type = "supporting"
            
        explanation = self._generate_mapping_explanation(model_rule, legal_rule, mapping_type)
        
        return ComplianceMapping(
            mapping_id=mapping_id,
            model_rule=model_rule,
            legal_rule=legal_rule,
            mapping_type=mapping_type,
            confidence=similarity,
            explanation=explanation
        )
        
    def _rules_conflict(self, model_rule: ModelRule, legal_rule: LegalRule) -> bool:
        """Check if model rule conflicts with legal rule."""
        # TODO: Implement conflict detection logic
        # This would analyze if the model rule violates the legal requirement
        return False
        
    def _generate_mapping_explanation(self, model_rule: ModelRule, legal_rule: LegalRule,
                                    mapping_type: str) -> str:
        """Generate human-readable explanation of the mapping."""
        explanations = {
            "direct": f"Model rule directly implements legal requirement: {legal_rule.premise}",
            "indirect": f"Model rule indirectly relates to legal requirement: {legal_rule.premise}",
            "conflicting": f"Model rule conflicts with legal requirement: {legal_rule.premise}",
            "supporting": f"Model rule supports compliance with legal requirement: {legal_rule.premise}"
        }
        
        return explanations.get(mapping_type, "Unknown mapping relationship")


class MappingReasoningEngine:
    """
    Main engine that orchestrates compliance mapping and reasoning.
    
    This serves as the automated auditor that systematically tests model logic
    against legal requirements from the knowledge base.
    """
    
    def __init__(self, knowledge_base: SymbolicKnowledgeBase):
        """
        Initialize the mapping and reasoning engine.
        
        Args:
            knowledge_base: The legal knowledge base to reason against
        """
        self.knowledge_base = knowledge_base
        self.mapper = ComplianceMapper()
        self.assessments: Dict[str, ComplianceAssessment] = {}
        
    def assess_compliance(self, system_id: str, model_rules: List[ModelRule],
                         system_type: str = "high_risk") -> ComplianceAssessment:
        """
        Perform comprehensive compliance assessment of an AI system.
        
        Args:
            system_id: Identifier for the AI system
            model_rules: Symbolic rules extracted from the model
            system_type: Type of AI system (determines applicable requirements)
            
        Returns:
            Complete compliance assessment
        """
        logger.info(f"Starting compliance assessment for system: {system_id}")
        
        # Get applicable legal requirements
        legal_rules = self.knowledge_base.query_requirements("general", system_type)
        
        # Map model rules to legal requirements
        mappings = self.mapper.map_rules_to_requirements(model_rules, legal_rules)
        
        # Identify violations
        violations = self._identify_violations(mappings)
        
        # Determine overall compliance status
        overall_status, confidence = self._determine_overall_status(mappings, violations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations, mappings)
        
        # Create assessment summary
        summary = self._generate_assessment_summary(overall_status, len(violations), len(mappings))
        
        assessment = ComplianceAssessment(
            system_id=system_id,
            overall_status=overall_status,
            confidence_score=confidence,
            mappings=mappings,
            violations=violations,
            recommendations=recommendations,
            assessment_summary=summary
        )
        
        self.assessments[system_id] = assessment
        logger.info(f"Compliance assessment completed for {system_id}: {overall_status.value}")
        
        return assessment
        
    def _identify_violations(self, mappings: List[ComplianceMapping]) -> List[ComplianceViolation]:
        """Identify compliance violations from rule mappings."""
        violations = []
        
        for mapping in mappings:
            if mapping.mapping_type == "conflicting":
                violation = ComplianceViolation(
                    violation_id=f"violation_{len(violations)}",
                    rule_id=mapping.model_rule.rule_id,
                    legal_requirement=mapping.legal_rule.premise,
                    violation_type="direct_conflict",
                    severity="critical",
                    description=f"Model rule conflicts with legal requirement: {mapping.explanation}",
                    suggested_remediation="Review and modify model training to align with legal requirements",
                    confidence=mapping.confidence
                )
                violations.append(violation)
                
        # TODO: Implement more sophisticated violation detection
        # - Missing required safeguards
        # - Prohibited practices
        # - Insufficient transparency measures
        # - Inadequate human oversight
        
        return violations
        
    def _determine_overall_status(self, mappings: List[ComplianceMapping],
                                violations: List[ComplianceViolation]) -> Tuple[ComplianceStatus, float]:
        """Determine overall compliance status and confidence."""
        if not mappings:
            return ComplianceStatus.UNKNOWN, 0.0
            
        # Count different types of mappings
        conflicting = len([m for m in mappings if m.mapping_type == "conflicting"])
        direct = len([m for m in mappings if m.mapping_type == "direct"])
        
        # Determine status based on violations and mappings
        if conflicting > 0:
            critical_violations = len([v for v in violations if v.severity == "critical"])
            if critical_violations > 0:
                return ComplianceStatus.NON_COMPLIANT, 0.8
            else:
                return ComplianceStatus.PARTIALLY_COMPLIANT, 0.6
        elif direct > len(mappings) * 0.7:
            return ComplianceStatus.COMPLIANT, 0.9
        else:
            return ComplianceStatus.REQUIRES_REVIEW, 0.5
            
    def _generate_recommendations(self, violations: List[ComplianceViolation],
                                mappings: List[ComplianceMapping]) -> List[str]:
        """Generate recommendations for improving compliance."""
        recommendations = []
        
        if violations:
            recommendations.append("Address identified compliance violations before deployment")
            
            critical_violations = [v for v in violations if v.severity == "critical"]
            if critical_violations:
                recommendations.append("Prioritize resolution of critical violations")
                
        # Add specific recommendations based on mapping analysis
        conflicting_mappings = [m for m in mappings if m.mapping_type == "conflicting"]
        if conflicting_mappings:
            recommendations.append("Review model training data and objectives to eliminate conflicting rules")
            
        indirect_mappings = [m for m in mappings if m.mapping_type == "indirect"]
        if len(indirect_mappings) > len(mappings) * 0.5:
            recommendations.append("Strengthen alignment between model behavior and legal requirements")
            
        if not recommendations:
            recommendations.append("Continue monitoring compliance during deployment")
            
        return recommendations
        
    def _generate_assessment_summary(self, status: ComplianceStatus, 
                                   violation_count: int, mapping_count: int) -> str:
        """Generate a human-readable assessment summary."""
        status_descriptions = {
            ComplianceStatus.COMPLIANT: "The AI system demonstrates strong compliance with applicable regulations.",
            ComplianceStatus.NON_COMPLIANT: "The AI system has significant compliance issues that must be addressed.",
            ComplianceStatus.PARTIALLY_COMPLIANT: "The AI system shows partial compliance but requires improvements.",
            ComplianceStatus.UNKNOWN: "Compliance status could not be determined with available information.",
            ComplianceStatus.REQUIRES_REVIEW: "The AI system requires manual review to determine compliance status."
        }
        
        base_summary = status_descriptions.get(status, "Unknown compliance status.")
        
        if violation_count > 0:
            base_summary += f" {violation_count} compliance violations were identified."
            
        base_summary += f" Analysis based on {mapping_count} rule mappings."
        
        return base_summary
        
    def compare_systems(self, system_ids: List[str]) -> Dict[str, Any]:
        """
        Compare compliance assessments across multiple systems.
        
        Args:
            system_ids: List of system identifiers to compare
            
        Returns:
            Comparative analysis results
        """
        if not all(sid in self.assessments for sid in system_ids):
            missing = [sid for sid in system_ids if sid not in self.assessments]
            raise ValueError(f"Missing assessments for systems: {missing}")
            
        comparison = {
            'systems_compared': len(system_ids),
            'status_distribution': {},
            'violation_analysis': {},
            'recommendations': []
        }
        
        # Analyze status distribution
        statuses = [self.assessments[sid].overall_status for sid in system_ids]
        for status in ComplianceStatus:
            comparison['status_distribution'][status.value] = statuses.count(status)
            
        # Analyze violations
        all_violations = []
        for sid in system_ids:
            all_violations.extend(self.assessments[sid].violations)
            
        comparison['violation_analysis'] = {
            'total_violations': len(all_violations),
            'by_severity': {},
            'common_issues': []
        }
        
        # Count violations by severity
        for severity in ['critical', 'major', 'minor', 'warning']:
            count = len([v for v in all_violations if v.severity == severity])
            comparison['violation_analysis']['by_severity'][severity] = count
            
        return comparison
        
    def export_assessment(self, system_id: str, output_path: str, format: str = 'json') -> None:
        """
        Export compliance assessment to file.
        
        Args:
            system_id: System identifier
            output_path: Output file path
            format: Export format ('json', 'html', 'pdf')
        """
        if system_id not in self.assessments:
            raise ValueError(f"No assessment found for system: {system_id}")
            
        assessment = self.assessments[system_id]
        
        if format == 'json':
            import json
            
            assessment_dict = {
                'system_id': assessment.system_id,
                'overall_status': assessment.overall_status.value,
                'confidence_score': assessment.confidence_score,
                'assessment_summary': assessment.assessment_summary,
                'violations': [
                    {
                        'violation_id': v.violation_id,
                        'severity': v.severity,
                        'description': v.description,
                        'suggested_remediation': v.suggested_remediation
                    }
                    for v in assessment.violations
                ],
                'recommendations': assessment.recommendations
            }
            
            with open(output_path, 'w') as f:
                json.dump(assessment_dict, f, indent=2)
                
        # TODO: Implement HTML and PDF export formats
        
        logger.info(f"Exported assessment for {system_id} to {output_path}")