import os
import subprocess
import platform
import sys
from dotenv import load_dotenv


def run_command(command: list[str] | str, use_shell=False):
    is_str_command = isinstance(command, str)
    print(f"Executing command: {command if is_str_command else ' '.join(command)}")

    is_windows = platform.system() == "Windows"
    effective_shell = use_shell or is_windows

    process = subprocess.run(
        command, capture_output=True, text=True, shell=effective_shell
    )

    if process.returncode != 0:
        print("--- ERROR ---")
        print("STDOUT:")
        print(process.stdout)
        print("STDERR:")
        print(process.stderr)
        return False
    else:
        print(process.stdout)
        if process.stderr:
            print("Warning/Info (stderr):")
            print(process.stderr)
        return True


def main():
    load_dotenv()

    resource_group = os.environ["RESOURCE_GROUP"]
    workspace = os.environ["WORKSPACE_NAME"]
    common_args = ["--resource-group", resource_group, "--workspace-name", workspace]

    print("--- Step 1: Registering data asset ---")
    if not run_command(
        [
            "az",
            "ml",
            "data",
            "create",
            "--name",
            "training-data",
            "--version",
            "1",
            "--path",
            "./data/train_cleaned.csv",
            "--type",
            "uri_file",
            *common_args,
        ]
    ):
        print("Checking if data asset already exists...")
        if run_command(
            [
                "az",
                "ml",
                "data",
                "show",
                "--name",
                "training-data",
                "--version",
                "1",
                *common_args,
            ]
        ):
            print("Data asset 'training-data:1' already exists. Skipping.")
        else:
            print("Failed for a reason other than the asset existing. Aborting.")
            sys.exit(1)

    print("\n--- Step 2: Creating compute cluster ---")
    if not run_command(
        [
            "az",
            "ml",
            "compute",
            "create",
            "--file",
            "./azure-ml/compute.yml",
            *common_args,
        ]
    ):
        print("Checking if compute cluster already exists...")
        if run_command(
            ["az", "ml", "compute", "show", "--name", "cpu-cluster", *common_args]
        ):
            print("Compute cluster 'cpu-cluster' already exists. Skipping.")
        else:
            print("Failed for a reason other than the cluster existing. Aborting.")
            sys.exit(1)

    print("\n--- Generating requirements.txt for Azure ML Environment ---")
    poetry_command = (
        "poetry export -f requirements.txt --output requirements.txt --without-hashes"
    )
    if not run_command(poetry_command.split(), use_shell=True):
        print("Failed to generate requirements.txt from poetry. Aborting.")
        sys.exit(1)
    print("requirements.txt generated successfully.")

    print("\n--- Step 3: Creating environment ---")
    if not run_command(
        [
            "az",
            "ml",
            "environment",
            "create",
            "--file",
            "./azure-ml/environment.yml",
            *common_args,
        ]
    ):
        print("Checking if environment already exists...")
        if run_command(
            [
                "az",
                "ml",
                "environment",
                "show",
                "--name",
                "mlops-env",
                "--version",
                "1",
                *common_args,
            ]
        ):
            print("Environment 'mlops-env:1' already exists. Skipping.")
        else:
            print("Failed for a reason other than the environment existing. Aborting.")
            sys.exit(1)

    print("\n--- Step 4: Submitting the training job ---")
    if not run_command(
        ["az", "ml", "job", "create", "--file", "./azure-ml/job.yml", *common_args]
    ):
        print("Failed to submit training job. Aborting.")
        sys.exit(1)

    print("\n--- Training job submitted successfully! ---")
    print("Check the Azure ML Studio for progress.")


if __name__ == "__main__":
    main()
