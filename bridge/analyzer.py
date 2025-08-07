"""
Network Analyzer

Analyzes neural network structures and activation patterns for rule extraction.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ActivationPattern:
    """Represents an activation pattern in the neural network."""
    layer_name: str
    neuron_indices: List[int]
    activation_threshold: float
    pattern_frequency: float
    associated_outputs: List[Any]


@dataclass
class LayerAnalysis:
    """Analysis results for a specific layer."""
    layer_name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    activation_statistics: Dict[str, float]
    important_neurons: List[int]
    dead_neurons: List[int]


class ActivationHook:
    """Hook for capturing layer activations during forward pass."""
    
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.activations = []
        
    def __call__(self, module, input, output):
        """Hook function called during forward pass."""
        # Store activation (detach to avoid gradient computation)
        if isinstance(output, torch.Tensor):
            self.activations.append(output.detach().cpu())
        elif isinstance(output, (list, tuple)):
            # Handle multiple outputs
            self.activations.append([o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output])
            
    def clear(self):
        """Clear stored activations."""
        self.activations = []
        
    def get_activations(self) -> List[torch.Tensor]:
        """Get all stored activations."""
        return self.activations


class NetworkAnalyzer:
    """
    Analyzes neural network architecture and behavior for rule extraction.
    """
    
    def __init__(self):
        """Initialize the network analyzer."""
        self.hooks: Dict[str, ActivationHook] = {}
        self.layer_analyses: Dict[str, LayerAnalysis] = {}
        
    def analyze_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of model architecture.
        
        Args:
            model: Neural network model to analyze
            
        Returns:
            Dictionary with architecture analysis results
        """
        logger.info("Starting architecture analysis")
        
        analysis = {
            'model_type': type(model).__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'layers': [],
            'complexity_metrics': {},
            'potential_bottlenecks': []
        }
        
        # Analyze each layer
        for name, module in model.named_modules():
            if self._is_leaf_module(module):
                layer_info = self._analyze_layer(name, module)
                analysis['layers'].append(layer_info)
                
        # Calculate complexity metrics
        analysis['complexity_metrics'] = self._calculate_complexity_metrics(model)
        
        # Identify potential bottlenecks
        analysis['potential_bottlenecks'] = self._identify_bottlenecks(analysis['layers'])
        
        logger.info(f"Architecture analysis complete. Found {len(analysis['layers'])} layers")
        return analysis
        
    def _is_leaf_module(self, module: nn.Module) -> bool:
        """Check if module is a leaf node (no children)."""
        return len(list(module.children())) == 0
        
    def _analyze_layer(self, name: str, module: nn.Module) -> Dict[str, Any]:
        """Analyze a specific layer."""
        layer_info = {
            'name': name,
            'type': type(module).__name__,
            'parameters': sum(p.numel() for p in module.parameters()),
            'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
        }
        
        # Add layer-specific information
        if isinstance(module, nn.Linear):
            layer_info.update({
                'input_features': module.in_features,
                'output_features': module.out_features,
                'has_bias': module.bias is not None
            })
        elif isinstance(module, nn.Conv2d):
            layer_info.update({
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding
            })
        elif isinstance(module, nn.BatchNorm2d):
            layer_info.update({
                'num_features': module.num_features,
                'eps': module.eps,
                'momentum': module.momentum
            })
            
        return layer_info
        
    def _calculate_complexity_metrics(self, model: nn.Module) -> Dict[str, Any]:
        """Calculate model complexity metrics."""
        metrics = {}
        
        # Count different types of layers
        layer_counts = defaultdict(int)
        for module in model.modules():
            if self._is_leaf_module(module):
                layer_counts[type(module).__name__] += 1
                
        metrics['layer_counts'] = dict(layer_counts)
        
        # Calculate depth
        max_depth = 0
        def calculate_depth(module, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            for child in module.children():
                calculate_depth(child, current_depth + 1)
                
        calculate_depth(model)
        metrics['max_depth'] = max_depth
        
        return metrics
        
    def _identify_bottlenecks(self, layers: List[Dict[str, Any]]) -> List[str]:
        """Identify potential bottlenecks in the architecture."""
        bottlenecks = []
        
        for layer in layers:
            # Check for layers with very few parameters
            if layer['parameters'] < 100 and layer['type'] in ['Linear', 'Conv2d']:
                bottlenecks.append(f"Layer {layer['name']} has very few parameters ({layer['parameters']})")
                
            # Check for dramatic size reductions
            if layer['type'] == 'Linear' and 'input_features' in layer:
                reduction_ratio = layer['input_features'] / layer['output_features']
                if reduction_ratio > 10:
                    bottlenecks.append(f"Layer {layer['name']} has high reduction ratio ({reduction_ratio:.1f})")
                    
        return bottlenecks
        
    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """
        Register hooks to capture activations from specific layers.
        
        Args:
            model: Neural network model
            layer_names: List of layer names to hook (if None, hooks all layers)
        """
        if layer_names is None:
            layer_names = [name for name, _ in model.named_modules() if self._is_leaf_module(_)]
            
        for name, module in model.named_modules():
            if name in layer_names:
                hook = ActivationHook(name)
                self.hooks[name] = hook
                module.register_forward_hook(hook)
                
        logger.info(f"Registered hooks for {len(self.hooks)} layers")
        
    def clear_hooks(self) -> None:
        """Clear all activation data from hooks."""
        for hook in self.hooks.values():
            hook.clear()
            
    def analyze_activations(self, model: nn.Module, data_sample: torch.Tensor) -> Dict[str, LayerAnalysis]:
        """
        Analyze activation patterns across layers.
        
        Args:
            model: Neural network model
            data_sample: Sample data to run through the model
            
        Returns:
            Dictionary mapping layer names to their analyses
        """
        logger.info("Starting activation analysis")
        
        # Ensure hooks are registered
        if not self.hooks:
            self.register_hooks(model)
            
        # Clear previous activations
        self.clear_hooks()
        
        # Run forward pass to capture activations
        model.eval()
        with torch.no_grad():
            _ = model(data_sample)
            
        # Analyze each layer's activations
        analyses = {}
        for layer_name, hook in self.hooks.items():
            if hook.activations:
                analysis = self._analyze_layer_activations(layer_name, hook.activations)
                analyses[layer_name] = analysis
                self.layer_analyses[layer_name] = analysis
                
        logger.info(f"Activation analysis complete for {len(analyses)} layers")
        return analyses
        
    def _analyze_layer_activations(self, layer_name: str, activations: List[torch.Tensor]) -> LayerAnalysis:
        """Analyze activations for a specific layer."""
        if not activations:
            return LayerAnalysis(
                layer_name=layer_name,
                layer_type="unknown",
                input_shape=(),
                output_shape=(),
                activation_statistics={},
                important_neurons=[],
                dead_neurons=[]
            )
            
        # Concatenate all activations
        all_activations = torch.cat([act.flatten(1) for act in activations], dim=0)
        
        # Calculate statistics
        stats = {
            'mean': float(torch.mean(all_activations)),
            'std': float(torch.std(all_activations)),
            'min': float(torch.min(all_activations)),
            'max': float(torch.max(all_activations)),
            'sparsity': float(torch.mean((all_activations == 0).float())),
            'negative_ratio': float(torch.mean((all_activations < 0).float()))
        }
        
        # Identify important and dead neurons
        neuron_means = torch.mean(all_activations, dim=0)
        neuron_stds = torch.std(all_activations, dim=0)
        
        # Important neurons: high mean or high variance
        important_threshold = torch.quantile(neuron_means, 0.9)
        variance_threshold = torch.quantile(neuron_stds, 0.9)
        important_neurons = torch.where(
            (neuron_means > important_threshold) | (neuron_stds > variance_threshold)
        )[0].tolist()
        
        # Dead neurons: consistently zero or very low activation
        dead_threshold = 0.01
        dead_neurons = torch.where(
            (neuron_means < dead_threshold) & (neuron_stds < dead_threshold)
        )[0].tolist()
        
        return LayerAnalysis(
            layer_name=layer_name,
            layer_type="analyzed",  # TODO: Determine actual layer type
            input_shape=activations[0].shape,
            output_shape=activations[0].shape,
            activation_statistics=stats,
            important_neurons=important_neurons,
            dead_neurons=dead_neurons
        )


class ActivationAnalyzer:
    """
    Specialized analyzer for identifying meaningful activation patterns.
    """
    
    def __init__(self, significance_threshold: float = 0.1):
        """
        Initialize the activation analyzer.
        
        Args:
            significance_threshold: Threshold for pattern significance
        """
        self.significance_threshold = significance_threshold
        self.patterns: List[ActivationPattern] = []
        
    def find_activation_patterns(self, layer_analyses: Dict[str, LayerAnalysis],
                               output_labels: torch.Tensor) -> List[ActivationPattern]:
        """
        Find significant activation patterns correlated with outputs.
        
        Args:
            layer_analyses: Analysis results for each layer
            output_labels: Ground truth labels for correlation analysis
            
        Returns:
            List of significant activation patterns
        """
        patterns = []
        
        for layer_name, analysis in layer_analyses.items():
            # Focus on important neurons
            for neuron_idx in analysis.important_neurons:
                pattern = self._analyze_neuron_pattern(
                    layer_name, neuron_idx, analysis, output_labels
                )
                if pattern and pattern.pattern_frequency > self.significance_threshold:
                    patterns.append(pattern)
                    
        self.patterns = patterns
        logger.info(f"Found {len(patterns)} significant activation patterns")
        return patterns
        
    def _analyze_neuron_pattern(self, layer_name: str, neuron_idx: int,
                              analysis: LayerAnalysis, output_labels: torch.Tensor) -> Optional[ActivationPattern]:
        """Analyze activation pattern for a specific neuron."""
        # TODO: Implement detailed neuron pattern analysis
        # This would involve:
        # 1. Extracting activation values for the neuron across samples
        # 2. Finding threshold values that correlate with output patterns
        # 3. Calculating pattern frequency and significance
        
        # Placeholder implementation
        return ActivationPattern(
            layer_name=layer_name,
            neuron_indices=[neuron_idx],
            activation_threshold=0.5,
            pattern_frequency=0.2,
            associated_outputs=[]
        )
        
    def generate_pattern_rules(self, patterns: List[ActivationPattern]) -> List[str]:
        """
        Convert activation patterns to logical rules.
        
        Args:
            patterns: List of activation patterns
            
        Returns:
            List of logical rule strings
        """
        rules = []
        
        for i, pattern in enumerate(patterns):
            # Convert pattern to logical rule
            conditions = []
            for neuron_idx in pattern.neuron_indices:
                conditions.append(f"{pattern.layer_name}_neuron_{neuron_idx} > {pattern.activation_threshold}")
                
            condition_str = " AND ".join(conditions)
            rule = f"IF ({condition_str}) THEN output_pattern_{i}"
            rules.append(rule)
            
        return rules
        
    def visualize_patterns(self, patterns: List[ActivationPattern]) -> Dict[str, Any]:
        """
        Generate visualization data for activation patterns.
        
        Args:
            patterns: List of activation patterns to visualize
            
        Returns:
            Dictionary with visualization data
        """
        viz_data = {
            'pattern_count': len(patterns),
            'layer_distribution': defaultdict(int),
            'frequency_distribution': [],
            'neuron_importance': defaultdict(int)
        }
        
        for pattern in patterns:
            viz_data['layer_distribution'][pattern.layer_name] += 1
            viz_data['frequency_distribution'].append(pattern.pattern_frequency)
            
            for neuron_idx in pattern.neuron_indices:
                viz_data['neuron_importance'][f"{pattern.layer_name}_{neuron_idx}"] += 1
                
        return dict(viz_data)