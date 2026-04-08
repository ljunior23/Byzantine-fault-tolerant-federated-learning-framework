import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsLogger:
    def __init__(self, log_dir: str = "results"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict] = []
        self.start_time = time.time()

    def log(self, metrics: Dict):
        metrics["timestamp"] = time.time() - self.start_time
        self.records.append(metrics)
        logger.debug(f"Metrics logged: round={metrics.get('round')}, "
                     f"strategy={metrics.get('strategy')}")

    def save_json(self, filename: str = "metrics.json"):
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2, default=str)
        logger.info(f"Metrics saved to {path}")
        return path

    def save_csv(self, filename: str = "metrics.csv"):
        if not self.records:
            return
        path = self.log_dir / filename
        fieldnames = set()
        for r in self.records:
            fieldnames.update(r.keys())
        fieldnames = sorted(fieldnames)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)
        logger.info(f"CSV saved to {path}")
        return path

    def get_summary(self) -> Dict:
        if not self.records:
            return {}
        rounds = [r.get("round", 0) for r in self.records]
        strategies = [r.get("strategy", "unknown") for r in self.records]
        flagged = [len(r.get("flagged_clients", [])) for r in self.records]
        from collections import Counter
        return {
            "total_rounds": len(self.records),
            "strategy_distribution": dict(Counter(strategies)),
            "avg_flagged_per_round": sum(flagged) / len(flagged) if flagged else 0,
            "total_runtime_s": round(self.records[-1]["timestamp"], 2),
        }
