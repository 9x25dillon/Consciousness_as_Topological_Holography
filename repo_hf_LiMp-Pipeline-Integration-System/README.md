---
# HF Repo Card Metadata (required to silence the warning)
license: apache-2.0
pipeline_tag: text-generation
tags:
  - ensemble-learning
  - consensus
  - uncertainty-estimation
  - rag
  - fastapi
  - research
library_name: fastapi
language:
  - en
# optional, shown on the card
pretty_name: LiMp Pipeline – Topological Consensus
short_description: >
  FastAPI microservice + algorithms for phase-coherence consensus, trinary
  quantization, and Cardy-style boundary energy proxies inspired by EFL/3^627.
---
# LiMp Pipeline – Topological Consensus Staging

This repo hosts:
- `api/`: a FastAPI microservice for consensus, hallucination risk, and RAG reranking.
- `consensus/`: practical proxies for phase coherence and Cardy-style boundary energy inspired by the EFL/3^627 framework.
- `experiments/`: notebooks for Dual-LLM Resonance, Trinary vs Binary quantization, and Observer Saturation.
- `paper/`: the formal LaTeX spec.

Mathematical reference: see `paper/EFL_3^627_Unified.pdf`.  <!-- EFL spec -->
