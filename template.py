from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

project_name = "house_price_prediction"

list_of_files = [
    ".github/workflows/.gitkeep",

    f"src/{project_name}/__init__.py",

    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",

    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",

    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",

    "notebook/.gitkeep",
    "artifacts/.gitkeep",

    "templates/index.html",
    "templates/home.html",

    "static/style.css",

    "app.py",
    "requirements.txt",
    "setup.py",
    "README.md",
    ".gitignore",
    "Dockerfile"
]

for file in list_of_files:

    file_path = Path(file)

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file only if it doesn't exist or is empty
    if not file_path.exists() or file_path.stat().st_size == 0:

        file_path.touch(exist_ok=True)

        logging.info(f"Created: {file_path}")

    else:

        logging.info(f"Already exists: {file_path}")