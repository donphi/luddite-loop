# 🔄 The Luddite Loop: Core–Periphery Analysis of UK Biobank Research  

> *How medical conservatism trapped 8,553 studies in just 50 features*  

---

# 🚧 **STATUS: PENDING – Dissertation Work in Progress** 🚧  

---

## 📖 Overview  
This repository contains the code, pipelines, and documentation for my MSc dissertation:  
**"The Luddite Loop: How Traditional Feature Selection Constrains Discovery in UK Biobank Research"** (University of Westminster, 2025).  

The project reveals a striking paradox: While the UK Biobank offers **~10,000 data fields** for exploration, analysis of **8,500+ published papers** shows that research has converged on just **50 core features (0.48%)** that appear in 96.6% of all studies. Meanwhile, **6,764 features remain completely unexplored**.

This "Luddite Loop" represents a self-reinforcing cycle where traditional statistical thinking and cognitive limitations prevent researchers from leveraging the full power of modern AI and big data approaches, leaving 99.5% of the UK Biobank's potential untapped.

For background and methodology, see my [📄 Preliminary Report](preliminary-report.pdf).

### Key Findings
- **Gini coefficient of 0.866**: Extreme inequality in feature usage
- **Top 50 features**: Cover 96.6% of all research papers
- **6,764 features**: Never used in any published study
- **Bootstrap validation (n=1,000)**: Confirms stable core-periphery structure (Jaccard = 0.970)

### 🔬 Current Work: Methodology Refinement
**Status**: Actively refining feature extraction pipeline to minimize false positives
- Re-evaluating NLP extraction accuracy
- Implementing additional validation layers
- Cross-referencing extracted features against UK Biobank field IDs
- Strengthening statistical validation framework

*Note: Preliminary findings are compelling but undergoing rigorous re-validation to ensure maximum scientific accuracy.*

---