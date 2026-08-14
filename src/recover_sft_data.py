import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

from huggingface_hub import whoami
from datasets import Dataset, DatasetInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy loggers
for noisy in ("httpx", "LiteLLM", "transformers.tokenization_utils_base"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)


def is_recent(timestamp_str: str, hours: int) -> bool:
    """Check if timestamp is within the last N hours."""
    try:
        # Parse timestamp from directory name (e.g., "2025-06-26T01:41:49-86700e19")
        dt_str = timestamp_str.split('-')[0] + '-' + timestamp_str.split('-')[1] + '-' + timestamp_str.split('-')[2]
        dt = datetime.fromisoformat(dt_str)
        cutoff = datetime.now() - timedelta(hours=hours)
        return dt > cutoff
    except Exception as e:
        logger.warning(f"Failed to parse timestamp {timestamp_str}: {e}")
        return False


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of messages."""
    messages = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    messages.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
    return messages


def process_nano_history(nano_dir: Path, hours_ago: int, model_filter: str) -> List[Dict[str, Any]]:
    """Process nano history directories and extract valid data."""
    results = []
    
    if not nano_dir.exists():
        logger.error(f"Nano directory {nano_dir} does not exist")
        return results
    
    # Get all directories in ~/.nano
    history_dirs = [d for d in nano_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(history_dirs)} history directories")
    
    for hist_dir in history_dirs:
        try:
            # Check if recent
            dir_name = hist_dir.name
            if not is_recent(dir_name, hours_ago):
                continue
            
            # Check required files exist
            metadata_file = hist_dir / "metadata.json"
            diff_file = hist_dir / "diff.txt"
            messages_file = hist_dir / "messages.jsonl"
            tools_file = hist_dir / "tools.json"
            
            if not all(f.exists() for f in [metadata_file, diff_file, messages_file, tools_file]):
                logger.warning(f"Missing required files in {hist_dir}")
                continue
            
            # Load metadata and check model
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            model = metadata.get('model', '')
            if model_filter not in model:
                continue
            
            # Check if diff is non-empty
            with open(diff_file, 'r') as f:
                diff_content = f.read().strip()
            
            if not diff_content:
                logger.info(f"Empty diff in {hist_dir}")
                continue
            
            # Load tools
            with open(tools_file, 'r') as f:
                tools = json.load(f)
            
            # Load messages
            messages = load_jsonl(messages_file)
            
            if not messages:
                logger.warning(f"No messages found in {hist_dir}")
                continue
            
            # Extract problem statement (usually first user message)
            problem_statement = ""
            for msg in messages:
                if msg.get('role') == 'user':
                    problem_statement = msg.get('content', '')
                    break
            
            # Create minimal record for SFT training
            result = {
                'messages': messages,
                'tools': tools,
                'reward': -1.0,  # Dummy reward value
                'problem_statement': problem_statement,
                'generated_diff': diff_content,
            }
            
            results.append(result)
            logger.info(f"✓ Found valid history: {hist_dir.name} (diff length: {len(diff_content)})")
            
        except Exception as e:
            logger.error(f"Error processing {hist_dir}: {e}")
            continue
    
    return results


def main():
    """Main function to recover nano data."""
    parser = argparse.ArgumentParser(description="Recover nano agent SFT data")
    parser.add_argument("--hours-ago", type=int, default=3, 
                        help="Time window for recovery in hours (default: 3)")
    parser.add_argument("--model-filter", type=str, default="gemini-2.5-flash",
                        help="Model name to filter for (default: gemini-2.5-flash)")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push dataset to HuggingFace Hub")
    
    args = parser.parse_args()
    
    logger.info(f"Starting nano data recovery...")
    logger.info(f"Parameters: hours_ago={args.hours_ago}, model_filter={args.model_filter}, push_to_hub={args.push_to_hub}")
    
    # Check HuggingFace login if pushing to hub
    if args.push_to_hub:
        try:
            whoami()
        except Exception:
            raise ValueError("Not logged in to HuggingFace. Please run 'huggingface-cli login' first.")
    
    # Process histories
    nano_dir = Path.home() / ".nano"
    recovered_data = process_nano_history(nano_dir, args.hours_ago, args.model_filter)
    
    logger.info(f"\nRecovered {len(recovered_data)} valid histories")
    
    if recovered_data:
        # Create dataset info with detailed description
        recovery_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_short = args.model_filter.split('/')[-1] if '/' in args.model_filter else args.model_filter
        
        info = DatasetInfo(
            description=f"""Recovered Nano agent SFT data from local histories.

## Recovery Details
- Recovery timestamp: {recovery_time}
- Time window: Last {args.hours_ago} hours
- Model filter: {args.model_filter}
- Total recovered histories: {len(recovered_data)}

## Dataset Structure
- `messages`: Full conversation history (list of message dicts)
- `tools`: Shell and navigation tools used (JSON)
- `reward`: -1.0 (dummy value for compatibility)
- `problem_statement`: Original task description
- `generated_diff`: Agent-generated solution diff

## Recovery Criteria
1. Non-empty diff files
2. Model contains "{args.model_filter}"
3. Created within {args.hours_ago} hours of recovery
4. Complete history with all required files

This dataset was recovered from interrupted or completed nano agent runs.
All histories contain successful code modifications as indicated by non-empty diffs.
Use ``only_passed=false`` when loading this recovered dataset for SFT training,
because recovered histories do not contain pass/fail evaluation results.
            """
        )
        
        # Create HuggingFace dataset
        logger.info("Creating HuggingFace dataset...")
        dataset = Dataset.from_list(recovered_data, info=info)
        
        dataset_name = f"ASSERT-KTH/Nano-SFT-Recovery-{model_short}"
        
        # Save dataset locally first
        local_path = f"data/{dataset_name.replace('/', '-')}"
        dataset.save_to_disk(local_path)
        logger.info(f"Dataset saved locally to {local_path}")
        
        # Push to HuggingFace Hub if requested
        if args.push_to_hub:
            logger.info(f"Pushing dataset to HuggingFace Hub: {dataset_name}")
            try:
                dataset.push_to_hub(
                    dataset_name,
                    commit_message=f"Recovered Nano SFT data - {len(recovered_data)} histories from {model_short}"
                )
                logger.info("Successfully pushed dataset to HuggingFace Hub")
                logger.info(f"Dataset URL: https://huggingface.co/datasets/{dataset_name}")
            except Exception as e:
                logger.error(f"Failed to push dataset to Hub: {e}")
                logger.info(f"Dataset is still available locally at {local_path}")
        
        # Print summary
        logger.info("\nSummary of recovered data:")
        logger.info(f"Total histories: {len(recovered_data)}")
        logger.info(f"Average diff length: {sum(len(r['generated_diff']) for r in recovered_data) / len(recovered_data):.0f} chars")
        logger.info(f"Average messages: {sum(len(r['messages']) for r in recovered_data) / len(recovered_data):.0f}")
        
    else:
        logger.warning("No valid histories found matching criteria")


if __name__ == "__main__":
    main()
