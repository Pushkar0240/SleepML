import json
from pathlib import Path

class HistoryManager:
    def __init__(self, output_dir="outputs/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, history, filename="history.json"):
        # history is a keras History object
        history_dict = history.history
        # Convert values to float for JSON serialization
        history_dict = {k: [float(i) for i in v] for k, v in history_dict.items()}
        
        with open(self.output_dir / filename, "w") as f:
            json.dump(history_dict, f, indent=4)
        print(f"History saved to {self.output_dir / filename}")
