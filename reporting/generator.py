"""
Compliance Report Generator

Generates comprehensive, human-readable compliance reports for stakeholders.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..mapping.engine import ComplianceAssessment, ComplianceViolation, ComplianceMapping

logger = logging.getLogger(__name__)


@dataclass
class ReportMetadata:
    """Metadata for compliance reports."""
    report_id: str
    generated_at: datetime
    system_id: str
    report_type: str
    version: str = "1.0"
    author: str = "AI Rosetta Stone Engine"


@dataclass 
class ReportSection:
    """Represents a section of the compliance report."""
    title: str
    content: str
    charts: List[Dict[str, Any]]
    subsections: List['ReportSection']


class ReportBuilder:
    """
    Builds structured compliance reports from assessment data.
    """
    
    def __init__(self):
        """Initialize the report builder."""
        self.sections: List[ReportSection] = []
        self.metadata: Optional[ReportMetadata] = None
        
    def create_executive_summary(self, assessment: ComplianceAssessment) -> ReportSection:
        """
        Create executive summary section.
        
        Args:
            assessment: Compliance assessment results
            
        Returns:
            Executive summary report section
        """
        # Generate key metrics
        total_rules = len(assessment.mappings)
        violations = len(assessment.violations)
        critical_violations = len([v for v in assessment.violations if v.severity == "critical"])
        
        # Create summary content
        content = f"""
        <h2>Executive Summary</h2>
        
        <div class="summary-box">
            <h3>Overall Compliance Status: {assessment.overall_status.value.upper()}</h3>
            <p><strong>Confidence Score:</strong> {assessment.confidence_score:.1%}</p>
            <p><strong>Assessment Summary:</strong> {assessment.assessment_summary}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric">
                <h4>Total Rules Analyzed</h4>
                <span class="metric-value">{total_rules}</span>
            </div>
            <div class="metric">
                <h4>Compliance Violations</h4>
                <span class="metric-value {'critical' if violations > 0 else ''}">{violations}</span>
            </div>
            <div class="metric">
                <h4>Critical Issues</h4>
                <span class="metric-value {'critical' if critical_violations > 0 else ''}">{critical_violations}</span>
            </div>
        </div>
        
        <div class="key-findings">
            <h4>Key Findings:</h4>
            <ul>
        """
        
        # Add key findings based on assessment
        if assessment.overall_status.value == "compliant":
            content += "<li>✅ System demonstrates strong regulatory compliance</li>"
        elif assessment.overall_status.value == "non_compliant":
            content += "<li>❌ System has critical compliance issues requiring immediate attention</li>"
        elif assessment.overall_status.value == "partially_compliant":
            content += "<li>⚠️ System shows partial compliance with areas for improvement</li>"
            
        if critical_violations > 0:
            content += f"<li>🚨 {critical_violations} critical violations identified</li>"
            
        if assessment.recommendations:
            content += f"<li>📋 {len(assessment.recommendations)} recommendations provided</li>"
            
        content += """
            </ul>
        </div>
        """
        
        return ReportSection(
            title="Executive Summary",
            content=content,
            charts=[],
            subsections=[]
        )
        
    def create_detailed_analysis(self, assessment: ComplianceAssessment) -> ReportSection:
        """
        Create detailed analysis section.
        
        Args:
            assessment: Compliance assessment results
            
        Returns:
            Detailed analysis report section
        """
        content = """
        <h2>Detailed Compliance Analysis</h2>
        
        <h3>Rule Mapping Analysis</h3>
        <p>The following analysis shows how AI model rules map to legal requirements:</p>
        
        <table class="mapping-table">
            <thead>
                <tr>
                    <th>Model Rule ID</th>
                    <th>Legal Requirement</th>
                    <th>Mapping Type</th>
                    <th>Confidence</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add mapping details
        for mapping in assessment.mappings:
            status_icon = {
                "direct": "✅",
                "indirect": "⚠️", 
                "conflicting": "❌",
                "supporting": "✅"
            }.get(mapping.mapping_type, "❓")
            
            content += f"""
                <tr class="mapping-{mapping.mapping_type}">
                    <td>{mapping.model_rule.rule_id}</td>
                    <td>{mapping.legal_rule.premise[:100]}...</td>
                    <td>{mapping.mapping_type.title()}</td>
                    <td>{mapping.confidence:.1%}</td>
                    <td>{status_icon}</td>
                </tr>
            """
            
        content += """
            </tbody>
        </table>
        """
        
        # Add violation analysis if any
        if assessment.violations:
            content += """
            <h3>Compliance Violations</h3>
            <div class="violations-section">
            """
            
            for violation in assessment.violations:
                severity_class = violation.severity.lower()
                content += f"""
                <div class="violation-item {severity_class}">
                    <h4>Violation: {violation.violation_id}</h4>
                    <p><strong>Severity:</strong> {violation.severity.upper()}</p>
                    <p><strong>Description:</strong> {violation.description}</p>
                    <p><strong>Legal Requirement:</strong> {violation.legal_requirement}</p>
                    <p><strong>Suggested Remediation:</strong> {violation.suggested_remediation}</p>
                    <p><strong>Confidence:</strong> {violation.confidence:.1%}</p>
                </div>
                """
                
            content += "</div>"
            
        return ReportSection(
            title="Detailed Analysis",
            content=content,
            charts=[],
            subsections=[]
        )
        
    def create_recommendations_section(self, assessment: ComplianceAssessment) -> ReportSection:
        """
        Create recommendations section.
        
        Args:
            assessment: Compliance assessment results
            
        Returns:
            Recommendations report section
        """
        content = """
        <h2>Recommendations</h2>
        <p>Based on the compliance analysis, the following recommendations are provided:</p>
        
        <div class="recommendations-list">
        """
        
        for i, recommendation in enumerate(assessment.recommendations, 1):
            content += f"""
            <div class="recommendation-item">
                <h4>Recommendation {i}</h4>
                <p>{recommendation}</p>
            </div>
            """
            
        content += "</div>"
        
        # Add next steps
        content += """
        <h3>Next Steps</h3>
        <ol class="next-steps">
            <li>Review and prioritize the recommendations above</li>
            <li>Implement necessary changes to address compliance violations</li>
            <li>Re-run compliance assessment to verify improvements</li>
            <li>Establish ongoing monitoring for continuous compliance</li>
        </ol>
        """
        
        return ReportSection(
            title="Recommendations",
            content=content,
            charts=[],
            subsections=[]
        )


class ComplianceReportGenerator:
    """
    Main generator for comprehensive compliance reports.
    
    Produces human-readable and machine-auditable compliance reports
    that can be reviewed by technical, legal, and business stakeholders.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the report generator.
        
        Args:
            template_dir: Directory containing report templates
        """
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.builder = ReportBuilder()
        
        # Initialize Jinja2 environment
        if self.template_dir.exists():
            self.jinja_env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        else:
            self.jinja_env = Environment(loader=FileSystemLoader("."))
            
    def generate_compliance_report(self, assessment: ComplianceAssessment,
                                 output_path: Path, format: str = "html") -> None:
        """
        Generate a comprehensive compliance report.
        
        Args:
            assessment: Compliance assessment results
            output_path: Path for the generated report
            format: Report format ('html', 'pdf', 'json')
        """
        logger.info(f"Generating compliance report for system: {assessment.system_id}")
        
        # Create report metadata
        metadata = ReportMetadata(
            report_id=f"report_{assessment.system_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            system_id=assessment.system_id,
            report_type="compliance_assessment"
        )
        
        # Build report sections
        sections = [
            self.builder.create_executive_summary(assessment),
            self.builder.create_detailed_analysis(assessment),
            self.builder.create_recommendations_section(assessment)
        ]
        
        # Generate visualizations
        charts = self._generate_charts(assessment)
        
        # Render report based on format
        if format == "html":
            self._generate_html_report(metadata, sections, charts, output_path)
        elif format == "pdf":
            self._generate_pdf_report(metadata, sections, charts, output_path)
        elif format == "json":
            self._generate_json_report(metadata, assessment, output_path)
        else:
            raise ValueError(f"Unsupported report format: {format}")
            
        logger.info(f"Compliance report generated: {output_path}")
        
    def _generate_charts(self, assessment: ComplianceAssessment) -> List[Dict[str, Any]]:
        """Generate charts for the compliance report."""
        charts = []
        
        # Compliance status pie chart
        status_counts = {
            "Compliant": len([m for m in assessment.mappings if m.mapping_type == "direct"]),
            "Conflicting": len([m for m in assessment.mappings if m.mapping_type == "conflicting"]),
            "Indirect": len([m for m in assessment.mappings if m.mapping_type == "indirect"]),
            "Supporting": len([m for m in assessment.mappings if m.mapping_type == "supporting"])
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.3,
            marker_colors=['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
        )])
        
        fig.update_layout(
            title="Rule Mapping Distribution",
            showlegend=True,
            width=500,
            height=400
        )
        
        charts.append({
            'id': 'mapping_distribution',
            'title': 'Rule Mapping Distribution',
            'figure': fig,
            'type': 'pie'
        })
        
        # Violation severity chart if violations exist
        if assessment.violations:
            severity_counts = {}
            for violation in assessment.violations:
                severity_counts[violation.severity] = severity_counts.get(violation.severity, 0) + 1
                
            fig = go.Figure(data=[go.Bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                marker_color=['#e74c3c', '#f39c12', '#f1c40f', '#3498db']
            )])
            
            fig.update_layout(
                title="Violations by Severity",
                xaxis_title="Severity",
                yaxis_title="Count",
                width=500,
                height=400
            )
            
            charts.append({
                'id': 'violation_severity',
                'title': 'Violations by Severity',
                'figure': fig,
                'type': 'bar'
            })
            
        return charts
        
    def _generate_html_report(self, metadata: ReportMetadata, sections: List[ReportSection],
                            charts: List[Dict[str, Any]], output_path: Path) -> None:
        """Generate HTML compliance report."""
        
        # Default HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Compliance Report - {{ metadata.system_id }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { border-bottom: 2px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }
                .summary-box { background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }
                .metric { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }
                .metric-value { font-size: 2em; font-weight: bold; display: block; }
                .metric-value.critical { color: #e74c3c; }
                .mapping-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .mapping-table th, .mapping-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                .mapping-table th { background-color: #f2f2f2; }
                .mapping-conflicting { background-color: #ffebee; }
                .mapping-direct { background-color: #e8f5e8; }
                .violation-item { margin: 20px 0; padding: 15px; border-radius: 8px; }
                .violation-item.critical { background-color: #ffebee; border-left: 4px solid #e74c3c; }
                .violation-item.major { background-color: #fff3e0; border-left: 4px solid #f39c12; }
                .recommendation-item { margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }
                .chart-container { margin: 20px 0; text-align: center; }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="header">
                <h1>AI Compliance Assessment Report</h1>
                <p><strong>System:</strong> {{ metadata.system_id }}</p>
                <p><strong>Generated:</strong> {{ metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                <p><strong>Report ID:</strong> {{ metadata.report_id }}</p>
            </div>
            
            {% for section in sections %}
                <div class="section">
                    {{ section.content | safe }}
                </div>
            {% endfor %}
            
            <h2>Visualizations</h2>
            {% for chart in charts %}
                <div class="chart-container">
                    <div id="{{ chart.id }}"></div>
                </div>
            {% endfor %}
            
            <script>
                {% for chart in charts %}
                    Plotly.newPlot('{{ chart.id }}', {{ chart.figure.to_json() | safe }});
                {% endfor %}
            </script>
            
            <div class="footer" style="margin-top: 50px; border-top: 1px solid #ddd; padding-top: 20px; color: #666;">
                <p><em>Report generated by AI Rosetta Stone Engine v{{ metadata.version }}</em></p>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            metadata=metadata,
            sections=sections,
            charts=charts
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    def _generate_pdf_report(self, metadata: ReportMetadata, sections: List[ReportSection],
                           charts: List[Dict[str, Any]], output_path: Path) -> None:
        """Generate PDF compliance report."""
        # TODO: Implement PDF generation using weasyprint or similar
        logger.warning("PDF generation not yet implemented. Use HTML format instead.")
        
    def _generate_json_report(self, metadata: ReportMetadata, assessment: ComplianceAssessment,
                            output_path: Path) -> None:
        """Generate JSON compliance report."""
        import json
        
        report_data = {
            'metadata': {
                'report_id': metadata.report_id,
                'generated_at': metadata.generated_at.isoformat(),
                'system_id': metadata.system_id,
                'report_type': metadata.report_type,
                'version': metadata.version
            },
            'assessment': {
                'system_id': assessment.system_id,
                'overall_status': assessment.overall_status.value,
                'confidence_score': assessment.confidence_score,
                'assessment_summary': assessment.assessment_summary,
                'total_mappings': len(assessment.mappings),
                'total_violations': len(assessment.violations),
                'recommendations': assessment.recommendations,
                'violations': [
                    {
                        'violation_id': v.violation_id,
                        'rule_id': v.rule_id,
                        'severity': v.severity,
                        'description': v.description,
                        'suggested_remediation': v.suggested_remediation,
                        'confidence': v.confidence
                    }
                    for v in assessment.violations
                ],
                'mappings': [
                    {
                        'mapping_id': m.mapping_id,
                        'mapping_type': m.mapping_type,
                        'confidence': m.confidence,
                        'explanation': m.explanation
                    }
                    for m in assessment.mappings
                ]
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
    def generate_summary_dashboard(self, assessments: List[ComplianceAssessment],
                                 output_path: Path) -> None:
        """
        Generate a summary dashboard for multiple assessments.
        
        Args:
            assessments: List of compliance assessments
            output_path: Path for the dashboard file
        """
        # TODO: Implement multi-system dashboard
        logger.info(f"Generating summary dashboard for {len(assessments)} systems")
        
    def export_audit_trail(self, assessment: ComplianceAssessment, output_path: Path) -> None:
        """
        Export detailed audit trail for regulatory review.
        
        Args:
            assessment: Compliance assessment
            output_path: Path for audit trail file
        """
        # TODO: Implement detailed audit trail export
        logger.info(f"Exporting audit trail for system: {assessment.system_id}")