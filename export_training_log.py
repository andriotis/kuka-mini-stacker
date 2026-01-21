#!/usr/bin/env python3
"""
Export training metrics from terminal output to a tabular text file.
The output format is designed to be easily parsed by LLMs for analysis.
"""

import re
import sys
from pathlib import Path
from datetime import datetime


def parse_terminal_output(terminal_text: str) -> list[dict]:
    """Parse training metrics from terminal output."""
    records = []
    current_record = {}
    
    lines = terminal_text.split('\n')
    
    for line in lines:
        # Skip progress bars and separators
        if '━' in line or line.strip().startswith('-') and '|' not in line:
            continue
            
        # Parse eval summary lines like: "Eval num_timesteps=250000, episode_reward=-87.36 +/- 25.68"
        eval_match = re.search(
            r'Eval num_timesteps=(\d+), episode_reward=([-\d.]+) \+/- ([\d.]+)',
            line
        )
        if eval_match:
            timestep = int(eval_match.group(1))
            # Start a new record or update existing one for this timestep
            if current_record.get('timesteps') != timestep:
                if current_record:
                    records.append(current_record)
                current_record = {'timesteps': timestep}
            current_record['eval_reward'] = float(eval_match.group(1 + 1))
            current_record['eval_reward_std'] = float(eval_match.group(2 + 1))
            continue
        
        # Parse table rows like: "|    mean_reward          | -87.4       |"
        table_match = re.search(r'\|\s*(\w+(?:/\w+)?)\s*\|\s*([-\d.e]+)\s*\|', line)
        if table_match:
            key = table_match.group(1).strip()
            value_str = table_match.group(2).strip()
            try:
                value = float(value_str)
            except ValueError:
                continue
            
            # Categorize the metric
            if 'eval/' in line or key in ['mean_reward', 'mean_ep_length']:
                prefix = 'eval_'
            elif 'rollout/' in line or key in ['ep_len_mean', 'ep_rew_mean']:
                prefix = 'rollout_'
            elif 'train/' in line or key in ['approx_kl', 'clip_fraction', 'entropy_loss', 
                                              'explained_variance', 'policy_gradient_loss',
                                              'value_loss', 'loss', 'n_updates']:
                prefix = 'train_'
            elif 'time/' in line or key in ['fps', 'iterations', 'time_elapsed', 'total_timesteps']:
                prefix = 'time_'
            else:
                prefix = ''
            
            # Handle special cases
            if key == 'total_timesteps' and prefix == 'time_':
                # This is from the time section, might update our timestep reference
                if not current_record:
                    current_record = {'timesteps': int(value)}
                continue
            elif key == 'std' and prefix == 'train_':
                key = 'policy_std'  # Rename to avoid confusion
                
            current_record[f'{prefix}{key}'] = value
            continue
        
        # Check for "New best mean reward!" indicator
        if 'New best mean reward!' in line:
            current_record['new_best'] = True
    
    # Don't forget the last record
    if current_record:
        records.append(current_record)
    
    return records


def merge_records_by_timestep(records: list[dict]) -> list[dict]:
    """Merge records that belong to the same timestep checkpoint."""
    merged = {}
    
    for record in records:
        ts = record.get('timesteps')
        if ts is None:
            continue
            
        # Round to nearest 5000 for grouping
        ts_key = round(ts / 5000) * 5000
        
        if ts_key not in merged:
            merged[ts_key] = {'timesteps': ts_key}
        
        # Merge all fields
        for key, value in record.items():
            if key != 'timesteps':
                merged[ts_key][key] = value
    
    return sorted(merged.values(), key=lambda x: x['timesteps'])


def format_as_table(records: list[dict]) -> str:
    """Format records as a clean tabular text file."""
    if not records:
        return "No training data found."
    
    # Define column order for readability - these are the primary metrics
    column_order = [
        'timesteps',
        'eval_reward',
        'eval_reward_std', 
        'rollout_ep_rew_mean',
        'rollout_ep_len_mean',
        'train_loss',
        'train_value_loss',
        'train_policy_gradient_loss',
        'train_entropy_loss',
        'train_explained_variance',
        'train_approx_kl',
        'train_clip_fraction',
        'train_policy_std',
        'train_n_updates',
        'time_fps',
        'new_best'
    ]
    
    # Columns to exclude (duplicates or less useful)
    exclude_columns = {
        'eval_mean_reward', 'eval_mean_ep_length',  # duplicates of eval_reward
        'clip_range', 'clip_range_vf', 'learning_rate',  # constant values
        'std',  # renamed to train_policy_std
        'time_iterations', 'time_time_elapsed',  # less useful for analysis
    }
    
    # Collect all columns that actually have data
    all_keys = set()
    for record in records:
        all_keys.update(record.keys())
    
    # Filter to columns that exist, maintaining order, excluding redundant ones
    columns = [col for col in column_order if col in all_keys and col not in exclude_columns]
    # Add any remaining columns not in our predefined order (except excluded ones)
    for key in sorted(all_keys):
        if key not in columns and key not in exclude_columns:
            columns.append(key)
    
    # Build output
    output_lines = []
    
    # Header
    output_lines.append("# PPO Training Log")
    output_lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"# Total checkpoints: {len(records)}")
    output_lines.append("#")
    output_lines.append("# Column descriptions:")
    output_lines.append("#   timesteps         - Training step count")
    output_lines.append("#   eval_reward       - Mean episode reward during evaluation")
    output_lines.append("#   eval_reward_std   - Std deviation of eval rewards")
    output_lines.append("#   rollout_ep_rew_mean - Mean episode reward during training rollouts")
    output_lines.append("#   rollout_ep_len_mean - Mean episode length during rollouts")
    output_lines.append("#   train_loss        - Overall training loss")
    output_lines.append("#   train_value_loss  - Value function loss")
    output_lines.append("#   train_policy_gradient_loss - Policy gradient loss")
    output_lines.append("#   train_entropy_loss - Entropy bonus (more negative = more exploration)")
    output_lines.append("#   train_explained_variance - How well value fn predicts returns (0-1)")
    output_lines.append("#   train_approx_kl   - KL divergence between old/new policy")
    output_lines.append("#   train_clip_fraction - Fraction of samples clipped by PPO")
    output_lines.append("#   train_policy_std  - Standard deviation of policy actions")
    output_lines.append("#   train_n_updates   - Number of gradient updates performed")
    output_lines.append("#   time_fps          - Training frames per second")
    output_lines.append("#   new_best          - Whether this was a new best reward")
    output_lines.append("#")
    output_lines.append("")
    
    # Calculate column widths
    col_widths = {}
    for col in columns:
        max_width = len(col)
        for record in records:
            val = record.get(col, '')
            if isinstance(val, float):
                val_str = f"{val:.6f}" if abs(val) < 0.01 else f"{val:.2f}"
            elif isinstance(val, bool):
                val_str = "YES" if val else ""
            else:
                val_str = str(val)
            max_width = max(max_width, len(val_str))
        col_widths[col] = max_width + 2
    
    # Header row
    header = "".join(col.ljust(col_widths[col]) for col in columns)
    output_lines.append(header)
    output_lines.append("-" * len(header))
    
    # Data rows
    for record in records:
        row_parts = []
        for col in columns:
            val = record.get(col, '')
            if isinstance(val, float):
                val_str = f"{val:.6f}" if abs(val) < 0.01 else f"{val:.2f}"
            elif isinstance(val, bool):
                val_str = "YES" if val else ""
            else:
                val_str = str(val) if val else ''
            row_parts.append(val_str.ljust(col_widths[col]))
        output_lines.append("".join(row_parts))
    
    # Summary statistics
    output_lines.append("")
    output_lines.append("# Summary Statistics")
    output_lines.append("#" + "=" * 50)
    
    if records:
        eval_rewards = [r['eval_reward'] for r in records if 'eval_reward' in r]
        if eval_rewards:
            output_lines.append(f"# Eval Reward - Start: {eval_rewards[0]:.2f}, End: {eval_rewards[-1]:.2f}")
            output_lines.append(f"# Eval Reward - Min: {min(eval_rewards):.2f}, Max: {max(eval_rewards):.2f}")
            output_lines.append(f"# Eval Reward - Improvement: {eval_rewards[-1] - eval_rewards[0]:.2f}")
        
        rollout_rewards = [r['rollout_ep_rew_mean'] for r in records if 'rollout_ep_rew_mean' in r]
        if rollout_rewards:
            output_lines.append(f"# Rollout Reward - Start: {rollout_rewards[0]:.2f}, End: {rollout_rewards[-1]:.2f}")
        
        best_count = sum(1 for r in records if r.get('new_best'))
        output_lines.append(f"# New best rewards found: {best_count}")
        output_lines.append(f"# Total timesteps: {records[-1].get('timesteps', 'N/A')}")
    
    return "\n".join(output_lines)


def main():
    # Read terminal output
    terminal_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    
    if terminal_file and terminal_file.exists():
        terminal_text = terminal_file.read_text()
    else:
        # Try to find the cursor terminal file
        cursor_terminal = Path.home() / ".cursor/projects/home-nikos-kuka-mini-stacker/terminals/1.txt"
        if cursor_terminal.exists():
            terminal_text = cursor_terminal.read_text()
        else:
            print("Error: No terminal file found. Provide path as argument.")
            sys.exit(1)
    
    # Parse and process
    records = parse_terminal_output(terminal_text)
    merged_records = merge_records_by_timestep(records)
    
    # Generate output
    output = format_as_table(merged_records)
    
    # Write to file
    output_file = Path("training_log.txt")
    output_file.write_text(output)
    print(f"Training log saved to: {output_file.absolute()}")
    print(f"Parsed {len(merged_records)} training checkpoints")


if __name__ == "__main__":
    main()
