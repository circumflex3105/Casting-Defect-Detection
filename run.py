import argparse
import subprocess
import sys

def parse_command_line_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--use-preprocessed", action="store_true")
  return parser.parse_args()

def execute_command(command_list):
  print(f"\nExecuting: {' '.join(command_list)}")
  result = subprocess.run(command_list)
  
  if result.returncode != 0:
    print(f"Command failed with exit code {result.returncode}")
    sys.exit(result.returncode)

def run_preprocessing_pipeline():
  execute_command(["python", "-m", "src.preprocessing.apply_clahe"])
  execute_command(["python", "-m", "src.preprocessing.extract_mask"])
  execute_command(["python", "-m", "src.visualization.visualize_preprocessing"])

def run_training_pipeline(use_preprocessed):
  command_list = ["python", "-m", "src.model.train"]
  if use_preprocessed:
    command_list.append("--use-preprocessed")
  execute_command(command_list)

def run_evaluation_pipeline(use_preprocessed):
  command_list = ["python", "-m", "src.evaluation.metrics"]
  if use_preprocessed:
    command_list.append("--use-preprocessed")
  execute_command(command_list)

def execute_master_pipeline():
  arguments = parse_command_line_arguments()
  
  execute_command(["python", "-m", "src.preprocessing.generate_splits"])
  if arguments.use_preprocessed:
    run_preprocessing_pipeline()
    
  run_training_pipeline(arguments.use_preprocessed)
  run_evaluation_pipeline(arguments.use_preprocessed)

if __name__ == "__main__":
  execute_master_pipeline()