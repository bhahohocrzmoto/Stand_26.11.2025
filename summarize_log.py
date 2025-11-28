import json
import argparse
from collections import Counter
from pathlib import Path

def summarize_log(file_path: Path):
    """Reads a PlotGeneration_Debug.json file and prints a summary of its contents."""
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing JSON file: {e}")
        return

    if not isinstance(logs, list):
        print("Error: JSON file does not contain a list of logs.")
        return

    # Use a counter to aggregate identical events
    event_counter = Counter()

    for entry in logs:
        # Create a consistent, hashable key for each event type
        event_key = (
            entry.get('stage', 'unknown'),
            entry.get('status', 'unknown'),
            entry.get('detail', 'unknown')
        )
        event_counter[event_key] += 1
    
    print(f"--- Summary of {file_path.name} ---")
    if not event_counter:
        print("Log is empty.")
        return

    # Print the aggregated results
    for (stage, status, detail), count in event_counter.items():
        print(f"\n- Event: [{stage}] / Status: [{status}]")
        print(f"  Detail: {detail}")
        print(f"  Occurrences: {count} times")
    
    print(f"\n--- End of Summary ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize a PlotGeneration_Debug.json log file.")
    parser.add_argument("log_file", help="Path to the PlotGeneration_Debug.json file.")
    args = parser.parse_args()
    
    summarize_log(Path(args.log_file))
