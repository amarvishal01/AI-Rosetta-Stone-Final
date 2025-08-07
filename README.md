# AI Rosetta Stone - Symbolic Knowledge Base Builder

**De-risking AI Deployment through Neuro-Symbolic Regulatory Compliance**

This repository implements the Symbolic Knowledge Base component of the AI Rosetta Stone engine, as described in the technical whitepaper. The module ingests legal articles from the EU AI Act and converts them into machine-readable logical predicates for automated compliance checking.

## 🎯 Overview

The AI Rosetta Stone addresses the critical gap between AI model performance and regulatory compliance. While existing explainability tools like LIME and SHAP provide feature attributions, they cannot demonstrate that a model's logic aligns with specific legal requirements. This module bridges that gap by:

1. **Entity Extraction**: Identifying key components in legal text (systems, requirements, persons, scopes)
2. **Relationship Extraction**: Finding connections between entities using NLP techniques  
3. **Predicate Generation**: Converting relationships into logical predicates for automated reasoning

## 🚀 Quick Start

### Demo Version (No Dependencies)
```bash
python3 demo_knowledge_base_builder.py
```

### Production Version (Requires spaCy)
```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the full version
python3 knowledge_base_builder.py
```

## 📋 Example Usage

**Input (Article 14 from EU AI Act):**
```
Article 14 (Human Oversight): High-risk AI systems shall be designed and developed in such a way that they can be effectively overseen by natural persons during the period in which the AI system is in use.
```

**Output (Logical Predicates):**
```python
[
  "requires(system_type='high-risk', component='human_oversight')",
  "is_a(human_oversight, 'effective')",
  "scope(human_oversight, period='in_use')"
]
```

## 🏗️ Architecture

The knowledge base builder follows a three-stage pipeline:

```
Legal Text → [Entity Extraction] → [Relationship Extraction] → [Predicate Generation] → Logical Predicates
```

### Core Components

- **EntityType Enum**: Categorizes extracted entities (system_type, component, person_type, etc.)
- **RelationType Enum**: Defines relationship types (requires, prohibits, is_a, scope, etc.)
- **Entity & Relationship Classes**: Data structures for extracted information
- **KnowledgeBaseBuilder**: Main processing class with NLP pipeline

### Supported Entity Types
- `SYSTEM_TYPE`: AI systems, high-risk systems
- `COMPONENT`: Human oversight, transparency, robustness
- `PERSON_TYPE`: Natural persons, humans, individuals
- `TIME_PERIOD`: Usage periods, lifecycles, continuous operation
- `ATTRIBUTE`: Effective, mandatory, prohibited

### Supported Relationships
- `REQUIRES`: System X requires component Y
- `OVERSEEN_BY`: System X overseen by person type Y  
- `IS_A`: Component X is attribute Y
- `SCOPE`: Component X applies during period Y

## 🔧 Technical Implementation

### Pattern-Based Extraction
The demo version uses regex patterns to identify:
- System types: `high-risk AI systems`, `AI systems`
- Requirements: `human oversight`, `transparency`, `robustness`
- Modal verbs: `shall`, `must`, `required to`
- Scope indicators: `during the period`, `in use`

### NLP-Enhanced Version
The production version leverages spaCy for:
- Named Entity Recognition (NER)
- Dependency parsing
- Advanced relationship extraction
- Confidence scoring

## 📊 Performance

**Demo Results on Article 14:**
- ✅ 8 entities extracted
- ✅ 7 relationships identified  
- ✅ 7 logical predicates generated
- ✅ 3/3 expected predicates matched (100% accuracy)

## 🔮 Integration with AI Rosetta Stone Engine

This module serves as the foundation for the complete AI Rosetta Stone system:

```
EU AI Act → [Symbolic Knowledge Base] → [Neuro-Symbolic Bridge] → [Mapping & Reasoning] → Compliance Report
```

The generated predicates can be:
- Queried by reasoning engines (Prolog, OWL)
- Mapped to neural network logic rules
- Used for automated compliance verification
- Extended with confidence scores and metadata

## 🛠️ Development

### File Structure
```
├── knowledge_base_builder.py      # Production version with spaCy
├── demo_knowledge_base_builder.py # Demo version (no dependencies)
├── requirements.txt               # Python dependencies
└── README.md                     # This documentation
```

### Dependencies
- **spaCy** (≥3.4.0): Advanced NLP processing
- **Python** (≥3.8): Core runtime
- **regex**: Enhanced pattern matching

### Testing
```python
from knowledge_base_builder import KnowledgeBaseBuilder

kb_builder = KnowledgeBaseBuilder()
result = kb_builder.process_article("Article text here...")

print(f"Entities: {len(result['entities'])}")
print(f"Relationships: {len(result['relationships'])}")  
print(f"Predicates: {result['predicates']}")
```

## 🎓 Academic Foundation

This implementation is based on the research presented in:
**"The AI Rosetta Stone Engine: De-risking AI Deployment through Neuro-Symbolic Regulatory Compliance"**

Key innovations:
- Direct mapping from legal text to logical predicates
- Neuro-symbolic bridge for model logic extraction
- Automated compliance verification framework
- Proactive compliance-by-design workflow

## 🚧 Future Enhancements

- [ ] Support for additional EU AI Act articles
- [ ] Integration with OWL/RDF ontologies  
- [ ] Confidence scoring and uncertainty quantification
- [ ] Multi-language support for EU regulations
- [ ] Integration with neural network analysis tools
- [ ] REST API for enterprise integration

## 📜 License

This project is part of the AI Rosetta Stone research initiative focused on advancing AI safety and regulatory compliance.

---

*For questions about the AI Rosetta Stone engine or enterprise deployment, please refer to the technical whitepaper or contact the research team.*
