# Byzantine-Fault-Tolerant Federated Learning Framework

**University of Michigan-Dearborn | CIS Trustworthy AI**

A modular, production-ready framework for evaluating Byzantine-robust federated learning defenses against realistic adversarial attacks on non-IID distributed data.

---

## Architecture

```
bft-fl/
├── src/
│   ├── server.py                   # BFT-FL server with adaptive aggregation
│   ├── client.py                   # FL client + non-IID partitioner
│   ├── aggregators/
│   │   ├── krum.py                 # Krum / Multi-Krum
│   │   ├── trimmed_mean.py         # Coordinate-wise Trimmed Mean
│   │   ├── geometric_median.py     # Geometric Median (Weiszfeld)
│   │   └── normalized.py          # Normalized Gradient Aggregation
│   ├── attacks/
│   │   └── byzantine_simulator.py  # 5 attack types + orchestrator
│   ├── detection/
│   │   └── detector.py            # Z-score + cosine + DBSCAN detection
│   ├── models/
│   │   └── cnn.py                 # MNIST + CIFAR-10 CNNs
│   └── utils/
│       └── metrics.py             # Metrics logging (JSON + CSV)
├── experiments/
│   └── run_experiments.py         # Full comparative evaluation runner
├── dashboard/
│   └── app.py                     # Streamlit real-time dashboard
├── tests/
│   └── test_core.py               # Pytest unit tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
pytest tests/ -v
```

### 3. Quick Experiment (subset)
```bash
python experiments/run_experiments.py \
    --dataset mnist \
    --rounds 20 \
    --num-clients 10 \
    --quick
```

### 4. Full Evaluation
```bash
python experiments/run_experiments.py \
    --dataset mnist \
    --rounds 50 \
    --num-clients 10
```

### 5. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### 6. Docker (Full Stack)
```bash
cd docker && docker-compose up --build
# Dashboard: http://localhost:8501
```

---

## Attack Suite

| Attack | Description | Evasion Difficulty |
|--------|-------------|-------------------|
| **Sign Flip** | Negate + scale gradients to maximize loss | Low |
| **Gradient Scaling** | Amplify gradients to dominate aggregation | Low |
| **Backdoor** | Model replacement via scaled poisoned gradients | Medium |
| **Random Noise** | Replace gradients with Gaussian noise | Low |
| **Adaptive** | Craft gradients to evade detection while harmful | High |
| **Label Flip** | Flip only output-layer gradients (targeted) | Medium |

---

## Defense Suite

| Defense | Mechanism | Optimal Against |
|---------|-----------|-----------------|
| **FedAvg** | Vanilla averaging (baseline) | Nothing |
| **Krum** | Select gradient closest to k neighbors | Scaling, sign flip |
| **Trimmed Mean** | Coordinate-wise trim + mean | Scaling outliers |
| **Geometric Median** | Minimize sum of distances | Sign flip, noise |
| **Normalized (NGA)** | Clip norm + average | Scaling |
| **Adaptive** | Auto-switch based on threat level | All attacks |

---

## Detection Layers

**Layer 1 — Z-Score Norm Filtering**
- Computes gradient L2 norms; flags clients with |z| > 2.5

**Layer 2 — Cosine Similarity Analysis**
- Computes cosine similarity to mean; flags negatively correlated clients

**Layer 3 — DBSCAN Clustering**
- Random-projected gradient vectors; DBSCAN noise points = suspected Byzantine

**Reputation System**
- Persistent reputation score per client, decays on flagging, recovers over time

---

## Adaptive Strategy Selection

```
threat_level = 0.4 × flagged_ratio + 0.3 × cosine_anomaly + 0.3 × zscore_ratio

threat_level ≥ 0.4 → Krum (strongest, slowest)
threat_level ≥ 0.1 → Trimmed Mean (balanced)
threat_level < 0.1 → Normalized (fastest, light defense)
```

---

## Expected Results

| Scenario | Target |
|----------|--------|
| Accuracy retention @ 30% malicious | > 85% |
| Adaptive vs FedAvg (sign flip, 30%) | +25–40% accuracy |
| Detection precision | > 80% |
| Aggregation latency (10 clients) | < 100ms |

---

## References

1. Blanchard et al. — *Machine Learning with Adversaries*, NeurIPS 2017 (Krum)
2. Yin et al. — *Byzantine-Robust Distributed Learning*, ICML 2018 (Trimmed Mean)
3. Chen et al. — *Distributed Statistical ML in Adversarial Settings*, SIGMETRICS 2017 (Geo Median)
4. Bhagoji et al. — *Analyzing Federated Learning through an Adversarial Lens*, ICML 2019
5. McMahan et al. — *Communication-Efficient Learning of Deep Networks*, AISTATS 2017

---

