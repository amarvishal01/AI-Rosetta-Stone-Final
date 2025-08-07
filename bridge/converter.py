"""
Symbolic Converter

Converts neural network patterns and behaviors into symbolic logical rules.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

import sympy as sp
from sympy.logic.boolalg import And, Or, Not, Implies
from sympy import symbols, Symbol

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of logical rules that can be generated."""
    IMPLICATION = "implication"
    EQUIVALENCE = "equivalence"
    CONSTRAINT = "constraint"
    CLASSIFICATION = "classification"


@dataclass
class LogicalPredicate:
    """Represents a logical predicate in symbolic form."""
    name: str
    variables: List[str]
    expression: str
    predicate_type: str  # 'atomic', 'compound', 'numerical'


@dataclass
class SymbolicRule:
    """Represents a complete symbolic rule."""
    rule_id: str
    premises: List[LogicalPredicate]
    conclusions: List[LogicalPredicate]
    rule_type: RuleType
    confidence: float
    symbolic_expression: str
    natural_language: str


class LogicalRuleBuilder:
    """
    Builds logical rules from various input formats and neural network patterns.
    """
    
    def __init__(self):
        """Initialize the logical rule builder."""
        self.symbol_table: Dict[str, Symbol] = {}
        self.predicates: Dict[str, LogicalPredicate] = {}
        self.rules: List[SymbolicRule] = []
        
    def create_predicate(self, name: str, variables: List[str], 
                        expression: str = None, predicate_type: str = "atomic") -> LogicalPredicate:
        """
        Create a logical predicate.
        
        Args:
            name: Name of the predicate
            variables: List of variable names
            expression: Optional symbolic expression
            predicate_type: Type of predicate
            
        Returns:
            Created logical predicate
        """
        if expression is None:
            if variables:
                expression = f"{name}({', '.join(variables)})"
            else:
                expression = name
                
        predicate = LogicalPredicate(
            name=name,
            variables=variables,
            expression=expression,
            predicate_type=predicate_type
        )
        
        self.predicates[name] = predicate
        return predicate
        
    def create_numerical_predicate(self, variable: str, operator: str, 
                                 threshold: float) -> LogicalPredicate:
        """
        Create a numerical comparison predicate.
        
        Args:
            variable: Variable name
            operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
            threshold: Threshold value
            
        Returns:
            Numerical predicate
        """
        name = f"{variable}_{operator}_{threshold}"
        expression = f"{variable} {operator} {threshold}"
        
        return self.create_predicate(
            name=name,
            variables=[variable],
            expression=expression,
            predicate_type="numerical"
        )
        
    def create_implication_rule(self, premises: List[LogicalPredicate],
                              conclusions: List[LogicalPredicate],
                              confidence: float = 1.0) -> SymbolicRule:
        """
        Create an implication rule (IF premises THEN conclusions).
        
        Args:
            premises: List of premise predicates
            conclusions: List of conclusion predicates
            confidence: Confidence score for the rule
            
        Returns:
            Created symbolic rule
        """
        rule_id = f"rule_impl_{len(self.rules)}"
        
        # Build symbolic expression
        if len(premises) == 1:
            premise_expr = premises[0].expression
        else:
            premise_expr = " AND ".join([p.expression for p in premises])
            
        if len(conclusions) == 1:
            conclusion_expr = conclusions[0].expression
        else:
            conclusion_expr = " AND ".join([c.expression for c in conclusions])
            
        symbolic_expression = f"({premise_expr}) → ({conclusion_expr})"
        
        # Generate natural language
        natural_language = self._generate_natural_language(premises, conclusions, RuleType.IMPLICATION)
        
        rule = SymbolicRule(
            rule_id=rule_id,
            premises=premises,
            conclusions=conclusions,
            rule_type=RuleType.IMPLICATION,
            confidence=confidence,
            symbolic_expression=symbolic_expression,
            natural_language=natural_language
        )
        
        self.rules.append(rule)
        return rule
        
    def _generate_natural_language(self, premises: List[LogicalPredicate],
                                 conclusions: List[LogicalPredicate],
                                 rule_type: RuleType) -> str:
        """Generate natural language description of a rule."""
        premise_text = self._predicates_to_natural_language(premises)
        conclusion_text = self._predicates_to_natural_language(conclusions)
        
        if rule_type == RuleType.IMPLICATION:
            return f"IF {premise_text} THEN {conclusion_text}"
        elif rule_type == RuleType.EQUIVALENCE:
            return f"{premise_text} IF AND ONLY IF {conclusion_text}"
        else:
            return f"{premise_text} IMPLIES {conclusion_text}"
            
    def _predicates_to_natural_language(self, predicates: List[LogicalPredicate]) -> str:
        """Convert predicates to natural language."""
        if len(predicates) == 1:
            return self._predicate_to_natural_language(predicates[0])
        else:
            pred_texts = [self._predicate_to_natural_language(p) for p in predicates]
            return " AND ".join(pred_texts)
            
    def _predicate_to_natural_language(self, predicate: LogicalPredicate) -> str:
        """Convert a single predicate to natural language."""
        if predicate.predicate_type == "numerical":
            # Parse numerical expression
            parts = predicate.expression.split()
            if len(parts) >= 3:
                var, op, val = parts[0], parts[1], parts[2]
                op_text = {
                    '>': 'is greater than',
                    '<': 'is less than',
                    '>=': 'is greater than or equal to',
                    '<=': 'is less than or equal to',
                    '==': 'equals',
                    '!=': 'does not equal'
                }.get(op, op)
                return f"{var} {op_text} {val}"
        
        # Default to expression
        return predicate.expression


class SymbolicConverter:
    """
    Main converter that transforms neural network patterns into symbolic rules.
    """
    
    def __init__(self):
        """Initialize the symbolic converter."""
        self.rule_builder = LogicalRuleBuilder()
        self.feature_mappings: Dict[str, str] = {}
        self.conversion_stats: Dict[str, int] = {
            'rules_created': 0,
            'predicates_created': 0,
            'failed_conversions': 0
        }
        
    def convert_decision_tree_rules(self, tree_rules: List[Dict[str, Any]],
                                  feature_names: List[str] = None) -> List[SymbolicRule]:
        """
        Convert decision tree rules to symbolic form.
        
        Args:
            tree_rules: List of decision tree rules
            feature_names: Optional feature names for better readability
            
        Returns:
            List of symbolic rules
        """
        logger.info(f"Converting {len(tree_rules)} decision tree rules to symbolic form")
        
        symbolic_rules = []
        
        for rule_data in tree_rules:
            try:
                # Parse rule condition and conclusion
                condition = rule_data.get('condition', '')
                conclusion = rule_data.get('conclusion', '')
                confidence = rule_data.get('confidence', 1.0)
                
                # Convert condition to predicates
                premises = self._parse_condition_to_predicates(condition, feature_names)
                
                # Convert conclusion to predicates
                conclusions = self._parse_conclusion_to_predicates(conclusion)
                
                # Create symbolic rule
                symbolic_rule = self.rule_builder.create_implication_rule(
                    premises=premises,
                    conclusions=conclusions,
                    confidence=confidence
                )
                
                symbolic_rules.append(symbolic_rule)
                self.conversion_stats['rules_created'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to convert rule: {e}")
                self.conversion_stats['failed_conversions'] += 1
                
        logger.info(f"Successfully converted {len(symbolic_rules)} rules to symbolic form")
        return symbolic_rules
        
    def _parse_condition_to_predicates(self, condition: str, 
                                     feature_names: List[str] = None) -> List[LogicalPredicate]:
        """Parse a condition string into logical predicates."""
        predicates = []
        
        # Handle compound conditions with AND/OR
        if ' AND ' in condition:
            sub_conditions = condition.split(' AND ')
        else:
            sub_conditions = [condition]
            
        for sub_condition in sub_conditions:
            sub_condition = sub_condition.strip()
            if not sub_condition or sub_condition == 'TRUE':
                continue
                
            # Parse numerical conditions (e.g., "feature_1 <= 0.5")
            for op in ['<=', '>=', '<', '>', '==', '!=']:
                if op in sub_condition:
                    parts = sub_condition.split(op)
                    if len(parts) == 2:
                        var = parts[0].strip()
                        val = float(parts[1].strip())
                        
                        # Map feature names if provided
                        if feature_names and var.startswith('feature_'):
                            try:
                                idx = int(var.split('_')[1])
                                if idx < len(feature_names):
                                    var = feature_names[idx]
                            except (ValueError, IndexError):
                                pass
                                
                        predicate = self.rule_builder.create_numerical_predicate(var, op, val)
                        predicates.append(predicate)
                        self.conversion_stats['predicates_created'] += 1
                        break
                        
        return predicates
        
    def _parse_conclusion_to_predicates(self, conclusion: str) -> List[LogicalPredicate]:
        """Parse a conclusion string into logical predicates."""
        predicates = []
        
        # Handle class predictions (e.g., "class = 1")
        if '=' in conclusion:
            parts = conclusion.split('=')
            if len(parts) == 2:
                attr = parts[0].strip()
                value = parts[1].strip()
                
                predicate = self.rule_builder.create_predicate(
                    name=f"{attr}_equals_{value}",
                    variables=[attr, value],
                    expression=f"{attr} = {value}",
                    predicate_type="classification"
                )
                predicates.append(predicate)
                self.conversion_stats['predicates_created'] += 1
                
        return predicates
        
    def convert_activation_patterns(self, patterns: List[Dict[str, Any]]) -> List[SymbolicRule]:
        """
        Convert neural network activation patterns to symbolic rules.
        
        Args:
            patterns: List of activation pattern dictionaries
            
        Returns:
            List of symbolic rules
        """
        logger.info(f"Converting {len(patterns)} activation patterns to symbolic form")
        
        symbolic_rules = []
        
        for pattern in patterns:
            try:
                # Extract pattern information
                layer_name = pattern.get('layer_name', 'unknown')
                neuron_indices = pattern.get('neuron_indices', [])
                threshold = pattern.get('activation_threshold', 0.5)
                outputs = pattern.get('associated_outputs', [])
                
                # Create premises for neuron activations
                premises = []
                for neuron_idx in neuron_indices:
                    neuron_var = f"{layer_name}_neuron_{neuron_idx}"
                    predicate = self.rule_builder.create_numerical_predicate(
                        neuron_var, '>', threshold
                    )
                    premises.append(predicate)
                    
                # Create conclusions for outputs
                conclusions = []
                for output in outputs:
                    conclusion_pred = self.rule_builder.create_predicate(
                        name=f"output_{output}",
                        variables=[],
                        expression=f"output = {output}",
                        predicate_type="classification"
                    )
                    conclusions.append(conclusion_pred)
                    
                if premises and conclusions:
                    symbolic_rule = self.rule_builder.create_implication_rule(
                        premises=premises,
                        conclusions=conclusions,
                        confidence=pattern.get('pattern_frequency', 1.0)
                    )
                    symbolic_rules.append(symbolic_rule)
                    self.conversion_stats['rules_created'] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to convert activation pattern: {e}")
                self.conversion_stats['failed_conversions'] += 1
                
        logger.info(f"Successfully converted {len(symbolic_rules)} activation patterns")
        return symbolic_rules
        
    def simplify_rules(self, rules: List[SymbolicRule]) -> List[SymbolicRule]:
        """
        Simplify and optimize symbolic rules.
        
        Args:
            rules: List of symbolic rules to simplify
            
        Returns:
            List of simplified rules
        """
        logger.info(f"Simplifying {len(rules)} symbolic rules")
        
        simplified_rules = []
        
        for rule in rules:
            try:
                # TODO: Implement rule simplification
                # - Combine similar rules
                # - Remove redundant conditions
                # - Optimize logical expressions
                
                # For now, just copy the rule
                simplified_rules.append(rule)
                
            except Exception as e:
                logger.warning(f"Failed to simplify rule {rule.rule_id}: {e}")
                
        logger.info(f"Rule simplification complete")
        return simplified_rules
        
    def export_to_prolog(self, rules: List[SymbolicRule], output_path: str) -> None:
        """
        Export symbolic rules to Prolog format.
        
        Args:
            rules: List of symbolic rules
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            f.write("% AI Rosetta Stone - Extracted Symbolic Rules\n")
            f.write("% Generated from neural network analysis\n\n")
            
            for rule in rules:
                # Convert to Prolog syntax
                prolog_rule = self._convert_to_prolog_syntax(rule)
                f.write(f"{prolog_rule}\n")
                
        logger.info(f"Exported {len(rules)} rules to Prolog format: {output_path}")
        
    def _convert_to_prolog_syntax(self, rule: SymbolicRule) -> str:
        """Convert a symbolic rule to Prolog syntax."""
        # Simplified conversion - more sophisticated logic needed for production
        premises_str = ", ".join([p.expression.replace(' ', '_') for p in rule.premises])
        conclusions_str = ", ".join([c.expression.replace(' ', '_') for c in rule.conclusions])
        
        return f"{conclusions_str} :- {premises_str}."
        
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """Get statistics about the conversion process."""
        return {
            **self.conversion_stats,
            'total_predicates': len(self.rule_builder.predicates),
            'total_rules': len(self.rule_builder.rules)
        }