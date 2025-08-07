"""
Neuro-Symbolic Bridge Extractor

Core component that extracts symbolic rules from trained neural networks.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRule:
    """Represents a symbolic rule extracted from a neural network."""
    rule_id: str
    condition: str
    conclusion: str
    confidence: float
    coverage: float
    layer_source: Optional[str] = None
    neuron_indices: Optional[List[int]] = None


@dataclass
class ExtractionConfig:
    """Configuration for rule extraction process."""
    method: str = "decision_tree"  # 'decision_tree', 'linear_approximation', 'activation_patterns'
    max_rules: int = 100
    min_confidence: float = 0.7
    min_coverage: float = 0.1
    sample_size: int = 10000
    feature_names: Optional[List[str]] = None


class RuleExtractor:
    """
    Base class for different rule extraction methods from neural networks.
    """
    
    def __init__(self, config: ExtractionConfig):
        """
        Initialize the rule extractor.
        
        Args:
            config: Configuration for the extraction process
        """
        self.config = config
        self.extracted_rules: List[ExtractedRule] = []
        
    def extract_rules(self, model: nn.Module, data_sample: torch.Tensor, 
                     target_sample: torch.Tensor = None) -> List[ExtractedRule]:
        """
        Extract symbolic rules from a neural network.
        
        Args:
            model: The trained neural network model
            data_sample: Sample input data for analysis
            target_sample: Optional target outputs for supervised extraction
            
        Returns:
            List of extracted symbolic rules
        """
        raise NotImplementedError("Subclasses must implement extract_rules method")
        
    def validate_rules(self, rules: List[ExtractedRule], validation_data: torch.Tensor) -> Dict[str, float]:
        """
        Validate extracted rules against validation data.
        
        Args:
            rules: List of extracted rules to validate
            validation_data: Validation dataset
            
        Returns:
            Dictionary with validation metrics
        """
        # TODO: Implement rule validation logic
        pass


class DecisionTreeExtractor(RuleExtractor):
    """
    Extracts rules using decision tree approximation of neural network behavior.
    """
    
    def extract_rules(self, model: nn.Module, data_sample: torch.Tensor,
                     target_sample: torch.Tensor = None) -> List[ExtractedRule]:
        """
        Extract rules using decision tree approximation.
        
        Args:
            model: The trained neural network model
            data_sample: Sample input data for analysis
            target_sample: Optional target outputs
            
        Returns:
            List of extracted symbolic rules
        """
        logger.info("Extracting rules using decision tree approximation")
        
        # Set model to evaluation mode
        model.eval()
        
        # Generate predictions from the neural network
        with torch.no_grad():
            if target_sample is None:
                predictions = model(data_sample)
                if predictions.dim() > 1:
                    # For classification, use argmax
                    target_sample = torch.argmax(predictions, dim=1)
                else:
                    # For regression, use predictions directly
                    target_sample = predictions
        
        # Convert to numpy for sklearn
        X = data_sample.numpy()
        y = target_sample.numpy()
        
        # Train decision tree to approximate the neural network
        dt = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=max(2, len(X) // 100),
            min_samples_leaf=max(1, len(X) // 200),
            random_state=42
        )
        
        dt.fit(X, y)
        
        # Extract rules from decision tree
        rules = self._extract_from_decision_tree(dt, X.shape[1])
        
        logger.info(f"Extracted {len(rules)} rules using decision tree method")
        return rules
        
    def _extract_from_decision_tree(self, dt: DecisionTreeClassifier, n_features: int) -> List[ExtractedRule]:
        """
        Extract symbolic rules from a trained decision tree.
        
        Args:
            dt: Trained decision tree classifier
            n_features: Number of input features
            
        Returns:
            List of extracted rules
        """
        rules = []
        
        # Get tree structure
        tree = dt.tree_
        feature_names = self.config.feature_names or [f"feature_{i}" for i in range(n_features)]
        
        def extract_path_rules(node_id, path_conditions, depth=0):
            """Recursively extract rules from tree paths."""
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node - create rule
                class_counts = tree.value[node_id][0]
                predicted_class = np.argmax(class_counts)
                confidence = class_counts[predicted_class] / np.sum(class_counts)
                coverage = np.sum(class_counts) / tree.n_samples
                
                if confidence >= self.config.min_confidence and coverage >= self.config.min_coverage:
                    condition = " AND ".join(path_conditions) if path_conditions else "TRUE"
                    conclusion = f"class = {predicted_class}"
                    
                    rule = ExtractedRule(
                        rule_id=f"dt_rule_{len(rules)}",
                        condition=condition,
                        conclusion=conclusion,
                        confidence=confidence,
                        coverage=coverage
                    )
                    rules.append(rule)
            else:
                # Internal node - continue down both branches
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                feature_name = feature_names[feature_idx]
                
                # Left branch (<=)
                left_condition = f"{feature_name} <= {threshold:.3f}"
                extract_path_rules(
                    tree.children_left[node_id],
                    path_conditions + [left_condition],
                    depth + 1
                )
                
                # Right branch (>)
                right_condition = f"{feature_name} > {threshold:.3f}"
                extract_path_rules(
                    tree.children_right[node_id],
                    path_conditions + [right_condition],
                    depth + 1
                )
        
        extract_path_rules(0, [])
        return rules[:self.config.max_rules]


class ActivationPatternExtractor(RuleExtractor):
    """
    Extracts rules based on neural network activation patterns.
    """
    
    def extract_rules(self, model: nn.Module, data_sample: torch.Tensor,
                     target_sample: torch.Tensor = None) -> List[ExtractedRule]:
        """
        Extract rules based on activation patterns in the network.
        
        Args:
            model: The trained neural network model
            data_sample: Sample input data for analysis
            target_sample: Optional target outputs
            
        Returns:
            List of extracted symbolic rules
        """
        logger.info("Extracting rules using activation pattern analysis")
        
        # TODO: Implement activation pattern analysis
        # 1. Hook into network layers to capture activations
        # 2. Identify significant activation patterns
        # 3. Correlate patterns with outputs
        # 4. Generate logical rules from patterns
        
        rules = []
        return rules


class LinearApproximationExtractor(RuleExtractor):
    """
    Extracts rules using linear approximation of network behavior in local regions.
    """
    
    def extract_rules(self, model: nn.Module, data_sample: torch.Tensor,
                     target_sample: torch.Tensor = None) -> List[ExtractedRule]:
        """
        Extract rules using local linear approximations.
        
        Args:
            model: The trained neural network model
            data_sample: Sample input data for analysis
            target_sample: Optional target outputs
            
        Returns:
            List of extracted symbolic rules
        """
        logger.info("Extracting rules using linear approximation method")
        
        # TODO: Implement linear approximation extraction
        # 1. Partition input space into regions
        # 2. Fit linear models in each region
        # 3. Convert linear models to logical rules
        # 4. Merge and simplify rules
        
        rules = []
        return rules


class NeuroSymbolicBridge:
    """
    Main class for the neuro-symbolic bridge that orchestrates rule extraction.
    
    This is the core innovation that extracts symbolic rules from neural networks
    and converts them into logical representations for compliance analysis.
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize the neuro-symbolic bridge.
        
        Args:
            config: Configuration for rule extraction
        """
        self.config = config or ExtractionConfig()
        self.extractors = self._initialize_extractors()
        self.extracted_rules: Dict[str, List[ExtractedRule]] = {}
        
    def _initialize_extractors(self) -> Dict[str, RuleExtractor]:
        """Initialize available rule extractors."""
        return {
            'decision_tree': DecisionTreeExtractor(self.config),
            'activation_patterns': ActivationPatternExtractor(self.config),
            'linear_approximation': LinearApproximationExtractor(self.config)
        }
        
    def extract_rules(self, model: nn.Module, data_sample: torch.Tensor,
                     method: str = None) -> List[ExtractedRule]:
        """
        Extract symbolic rules from a neural network model.
        
        This is the main method that converts opaque neural network logic
        into transparent symbolic rules that can be audited for compliance.
        
        Args:
            model: The trained neural network model to analyze
            data_sample: Representative sample of input data
            method: Extraction method to use (if None, uses config default)
            
        Returns:
            List of extracted symbolic rules
        """
        method = method or self.config.method
        
        if method not in self.extractors:
            raise ValueError(f"Unknown extraction method: {method}")
            
        logger.info(f"Starting rule extraction using method: {method}")
        
        extractor = self.extractors[method]
        rules = extractor.extract_rules(model, data_sample)
        
        # Store extracted rules
        model_id = f"model_{id(model)}"
        self.extracted_rules[model_id] = rules
        
        logger.info(f"Successfully extracted {len(rules)} rules from neural network")
        return rules
        
    def analyze_model_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze the architecture of a neural network model.
        
        Args:
            model: The neural network model to analyze
            
        Returns:
            Dictionary with architecture analysis results
        """
        analysis = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'layers': [],
            'model_type': type(model).__name__
        }
        
        # Analyze each layer
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = {
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                
                # Add layer-specific information
                if isinstance(module, nn.Linear):
                    layer_info['input_features'] = module.in_features
                    layer_info['output_features'] = module.out_features
                elif isinstance(module, nn.Conv2d):
                    layer_info['in_channels'] = module.in_channels
                    layer_info['out_channels'] = module.out_channels
                    layer_info['kernel_size'] = module.kernel_size
                
                analysis['layers'].append(layer_info)
        
        return analysis
        
    def generate_rule_report(self, rules: List[ExtractedRule]) -> Dict[str, Any]:
        """
        Generate a comprehensive report about extracted rules.
        
        Args:
            rules: List of extracted rules
            
        Returns:
            Dictionary with rule analysis report
        """
        if not rules:
            return {'error': 'No rules provided for analysis'}
            
        # Calculate statistics
        confidences = [rule.confidence for rule in rules]
        coverages = [rule.coverage for rule in rules]
        
        report = {
            'total_rules': len(rules),
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'average_coverage': np.mean(coverages),
            'total_coverage': sum(coverages),
            'high_confidence_rules': len([r for r in rules if r.confidence > 0.8]),
            'rule_complexity': self._analyze_rule_complexity(rules)
        }
        
        return report
        
    def _analyze_rule_complexity(self, rules: List[ExtractedRule]) -> Dict[str, Any]:
        """
        Analyze the complexity of extracted rules.
        
        Args:
            rules: List of extracted rules
            
        Returns:
            Dictionary with complexity metrics
        """
        complexities = []
        
        for rule in rules:
            # Count conditions in rule (simple heuristic)
            condition_count = rule.condition.count('AND') + rule.condition.count('OR') + 1
            complexities.append(condition_count)
            
        return {
            'average_conditions_per_rule': np.mean(complexities),
            'max_conditions_per_rule': np.max(complexities),
            'simple_rules': len([c for c in complexities if c <= 2]),
            'complex_rules': len([c for c in complexities if c > 5])
        }
        
    def export_rules(self, rules: List[ExtractedRule], output_path: Path, 
                    format: str = 'json') -> None:
        """
        Export extracted rules to a file.
        
        Args:
            rules: List of rules to export
            output_path: Path for the output file
            format: Export format ('json', 'csv', 'prolog')
        """
        import json
        import csv
        
        if format == 'json':
            rule_dicts = [
                {
                    'rule_id': rule.rule_id,
                    'condition': rule.condition,
                    'conclusion': rule.conclusion,
                    'confidence': rule.confidence,
                    'coverage': rule.coverage
                }
                for rule in rules
            ]
            
            with open(output_path, 'w') as f:
                json.dump(rule_dicts, f, indent=2)
                
        elif format == 'csv':
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Rule ID', 'Condition', 'Conclusion', 'Confidence', 'Coverage'])
                for rule in rules:
                    writer.writerow([
                        rule.rule_id, rule.condition, rule.conclusion,
                        rule.confidence, rule.coverage
                    ])
                    
        elif format == 'prolog':
            with open(output_path, 'w') as f:
                for rule in rules:
                    # Convert to Prolog-like format
                    f.write(f"rule({rule.rule_id}, '{rule.condition}', '{rule.conclusion}', {rule.confidence}).\n")
        
        logger.info(f"Exported {len(rules)} rules to {output_path} in {format} format")