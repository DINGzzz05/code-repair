import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from hydra.core.config_store import ConfigStore
from huggingface_hub import whoami
from datasets import Dataset, DatasetInfo
from omegaconf import OmegaConf

from src.data.swe_gym import SWE_GYM_HOLDOUT_RATIO, get_swe_gym_holdout_dataset
from src.agents.nano_agent import _process_one, NanoConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy loggers
for noisy in ("httpx", "LiteLLM", "transformers.tokenization_utils_base"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

@dataclass
class CurationConfig:
    # Dataset configuration
    input_dataset_name: str = "SWE-Gym/SWE-Gym-Lite"
    curation_ratio: float = SWE_GYM_HOLDOUT_RATIO
    dataset_version: str = "v1.0"
    push_to_hub: bool = False
    
    # Rollout configuration
    num_rollouts_per_problem: int = 8
    timeout: int = 120
    max_workers: int = 4  # ThreadPoolExecutor max workers
    max_problems: Optional[int] = None  # Maximum number of problems to process (for testing)
    

def get_output_dataset_name(curation_config: CurationConfig, agent_config: NanoConfig) -> str:
    """Generate output dataset name based on input dataset and model."""
    # Extract dataset short name (e.g., "SWE-Gym-Lite" from "SWE-Gym/SWE-Gym-Lite")
    input_short = curation_config.input_dataset_name.split('/')[-1]
    
    model_name = agent_config.model.split('/')[-1] or "local"
    
    return f"ASSERT-KTH/Nano-SFT-{input_short}-{model_name}"


@dataclass
class Config:
    curation: CurationConfig = field(default_factory=CurationConfig)
    agent: NanoConfig = field(default_factory=NanoConfig)

# Register the config schema
cs = ConfigStore.instance()
cs.store(name="base_curation_config", node=Config, group="")

def process_one(problem_data: dict[str, Any], config: NanoConfig, dataset_name: str) -> dict[str, Any]:
    """
    Run one Nano rollout and attach SWE-Gym metadata to the result.
    """
    result = _process_one(problem_data, config, dataset_name)

    result["instance_id"] = problem_data["instance_id"]
    result["problem_statement"] = problem_data["problem_statement"]
    result["repo"] = problem_data["repo"]
    result["base_commit"] = problem_data["base_commit"]
    result["oracle_diff"] = problem_data["patch"]
    result["oracle_test_diff"] = problem_data["test_patch"]

    return result


@hydra.main(version_base="1.1", config_path="conf", config_name="curation_config")
def main(cfg: Config) -> None:
    # Strict isolation: the SFT holdout partition and the GRPO repo-repair
    # partition must use the same ratio, otherwise the partitions overlap.
    if cfg.curation.curation_ratio != SWE_GYM_HOLDOUT_RATIO:
        raise ValueError(
            f"curation_ratio ({cfg.curation.curation_ratio}) must equal "
            f"SWE_GYM_HOLDOUT_RATIO ({SWE_GYM_HOLDOUT_RATIO}) so that the SFT "
            "holdout and GRPO repo-repair partitions stay strictly disjoint. "
            "Change SWE_GYM_HOLDOUT_RATIO in src/data/swe_gym.py if a different "
            "ratio is needed."
        )

    # Check HuggingFace login if pushing to hub
    if cfg.curation.push_to_hub:
        try:
            whoami()
        except Exception:
            raise ValueError("Not logged in to HuggingFace. Please run 'huggingface-cli login' first.")
    
    # Load SWE-Gym dataset
    logger.info("Loading SWE-Gym dataset...")
    dataset = get_swe_gym_holdout_dataset(cfg.curation.input_dataset_name, cfg.curation.curation_ratio)
    
    # Limit dataset size for testing
    if cfg.curation.max_problems:
        dataset = dataset.select(range(min(cfg.curation.max_problems, len(dataset))))
        logger.info(f"Limited to {len(dataset)} problems for testing")
    
    logger.info(f"Processing {len(dataset)} problems from SWE-Gym-Lite")
    
    # Create all rollout tasks upfront
    all_rollout_tasks = []
    for problem_data in dataset:
        for _ in range(cfg.curation.num_rollouts_per_problem):
            all_rollout_tasks.append(dict(problem_data))
    
    logger.info(f"Created {len(all_rollout_tasks)} total rollout tasks")
    
    # Process all rollouts using single ThreadPoolExecutor
    all_solutions = []
    completed_count = 0
    
    # Convert OmegaConf to NanoConfig dataclass like train_grpo.py does
    agent_config = NanoConfig(**OmegaConf.to_container(cfg.agent, resolve=True))
    
    with ThreadPoolExecutor(max_workers=cfg.curation.max_workers) as executor:
        futures = [executor.submit(process_one, task, agent_config, cfg.curation.input_dataset_name) for task in all_rollout_tasks]
        
        for future in as_completed(futures):
            completed_count += 1
            try:
                result = future.result(timeout=cfg.curation.timeout)
                solution = {
                    "instance_id": result["instance_id"],
                    "problem_statement": result["problem_statement"],
                    "repo": result["repo"],
                    "base_commit": result["base_commit"],
                    "oracle_diff": result["oracle_diff"],
                    "oracle_test_diff": result["oracle_test_diff"],
                    "generated_diff": result["generated_diff"],
                    "messages": result["prompt"] + result["completion"],
                    "tools": result["tools"]
                }
                all_solutions.append(solution)
                logger.info(f"[{completed_count}/{len(all_rollout_tasks)}] Completed rollout for {solution['instance_id']}")
            except Exception as e:
                logger.error(f"[{completed_count}/{len(all_rollout_tasks)}] Rollout failed: {e}")
                # Skip failed rollouts
                continue
    
    logger.info(f"Curation completed: {len(all_solutions)} solutions from {len(dataset)} problems")
    
    if not all_solutions:
        logger.warning("No solutions found! Check your model setup.")
        return
    
    # Create HuggingFace dataset
    logger.info("Creating HuggingFace dataset...")
    
    # Create dataset info with detailed description like in stack.py
    info = DatasetInfo(
        description=f"""SFT data generated by Nano, a terminal-based coding agent that uses tools like shell 
to navigate repositories and solve software engineering problems from the SWE-Gym dataset.

## Dataset Structure
- `instance_id`: Problem identifier from SWE-Gym
- `problem_statement`: Description of the coding problem
- `repo`: Repository information
- `base_commit`: Base commit hash
- `oracle_diff`: Ground truth patch diff
- `oracle_test_diff`: Ground truth test diff
- `generated_diff`: Agent-generated solution diff
- `messages`: Agent conversation with problem and solution
- `tools`: Shell and navigation tools used by the agent

## Generation Process
- {cfg.curation.num_rollouts_per_problem} rollouts per problem using Nano agent
- Curation ratio: {cfg.curation.curation_ratio} (sampling {cfg.curation.curation_ratio * 100:.0f}% of the dataset)
- All rollouts are retained; pass/fail filtering is applied when the dataset is loaded for SFT training via ``only_passed``
- When loading for training, instances with more than 4 passing rollouts keep only the 4 shortest trajectories
- Each passing trajectory gets a static weight ``sample_weight`` computed from the pass rate of its instance
  (``(num_total_rollouts / num_passed_rollouts) ** 0.5`` by default); low pass rate trajectories are up-weighted
  during SFT training, with the strength controlled by ``sample_weight_exponent``
- Generated with temperature {cfg.agent.temperature}, top-p {cfg.agent.top_p}
        """
    )
    
    curated_dataset = Dataset.from_list(all_solutions, info=info)
    
    # Save dataset locally first
    output_dataset_name = get_output_dataset_name(cfg.curation, cfg.agent)
    local_path = f"data/{output_dataset_name.replace('/', '-')}-{cfg.curation.dataset_version}"
    curated_dataset.save_to_disk(local_path)
    logger.info(f"Dataset saved locally to {local_path}")
    
    # Push to HuggingFace Hub if requested
    if cfg.curation.push_to_hub:
        logger.info(f"Pushing dataset to HuggingFace Hub: {output_dataset_name}")
        try:
            curated_dataset.push_to_hub(
                output_dataset_name,
                commit_message=f"Curated Nano SFT data v{cfg.curation.dataset_version} with {len(all_solutions)} solutions"
            )
            logger.info("Successfully pushed dataset to HuggingFace Hub")
            logger.info(f"Dataset URL: https://huggingface.co/datasets/{output_dataset_name}")
        except Exception as e:
            logger.error(f"Failed to push dataset to Hub: {e}")
            logger.info(f"Dataset is still available locally at {local_path}")
    
    logger.info("Data curation completed successfully!")


if __name__ == "__main__":
    main()
