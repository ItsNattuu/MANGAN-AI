# MANGAN-AI

### AI-Powered Manganese Exploration & Mining Decision-Support System

MANGAN-AI is an intelligent mineral exploration and mining decision-support platform designed to assist in identifying and prioritizing promising manganese exploration targets.

The system combines satellite-derived/geospatial data, machine learning, mining feasibility analysis, and agentic AI to transform large volumes of exploration data into actionable exploration priorities.

> **MANGAN-AI does not claim to directly prove the presence or quantity of underground manganese from satellite imagery. Instead, it identifies high-prospectivity targets that can be prioritized for geological field validation and further exploration.**

---

## Problem

Manganese exploration over large geographical areas can require significant time and resources.

The challenge is to determine:

- Which locations show promising manganese-related characteristics?
- Which candidate locations are practically suitable for exploration?
- Which targets should be investigated first?
- What factors contribute to the final prioritization?

MANGAN-AI addresses these challenges through a multi-layer AI pipeline.

---

# System Architecture

```text
                    MANGAN-AI
                         |
                         v
              +---------------------+
              |       LAYER 1       |
              | Satellite Processing|
              +----------+----------+
                         |
                         v
              +---------------------+
              |       LAYER 2       |
              | ML Prospectivity    |
              | Model               |
              +----------+----------+
                         |
                         v
              +---------------------+
              |       LAYER 3       |
              | Mining Feasibility  |
              | & Constraints       |
              +----------+----------+
                         |
                         v
              +---------------------+
              |       LAYER 4       |
              | Agentic AI          |
              | Decision Support    |
              +----------+----------+
                         |
                         v
                  Final Target
                  Prioritization
