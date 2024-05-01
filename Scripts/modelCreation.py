import os
import subprocess

import os


def delete_directory(directory_path):
    try:
        # Delete directory on Windows
        if subprocess.os.name == 'nt':
            subprocess.check_call(["rmdir", "/s", "/q", directory_path], shell=True)
        # Delete directory on Unix-like systems (Linux, macOS)
        else:
            subprocess.check_call(["rm", "-rf", directory_path])
        print(f"Directory '{directory_path}' deleted successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def create_folder(folder_name):
    try:
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully!")

        return True
    except OSError as e:
        print(f"Error: {e}")
        return False


def install_gitpython():
    try:
        subprocess.check_call(["pip", "install", "gitpython"])
        print("GitPython installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def clone_repository(repo_url, destination):
    import git
    try:
        git.Repo.clone_from(repo_url, destination)
        print("Repository cloned successfully!")
        return True
    except git.GitCommandError as e:
        print(f"Error: {e}")
        return False


def navigate_to_folder(folder_path):
    try:
        os.chdir(folder_path)
        print(f"Changed directory to: {os.getcwd()}")
    except OSError as e:
        print(f"Error: {e}")


def setup_venv():
    try:
        subprocess.check_call(["python", "-m", "venv", "venv"])
        print("Virtual environment created!")

        # Activate the virtual environment on Windows
        if subprocess.os.name == 'nt':
            activate_script = ".\\venv\\Scripts\\activate.bat"
        # Activate the virtual environment on Unix-based systems
        else:
            permission_script = "chmod +x ./venv/bin/activate"
            activate_script = "source venv/bin/activate"
            subprocess.check_call(permission_script, shell=True)

        # Activate the virtual environment
        subprocess.check_call([activate_script])
        print("Virtual environment activated!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def install_requirements():
    try:
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        subprocess.check_call(["pip", "install", "comet_ml"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def download_file():
    print("Downloading Started")
    import requests
    try:
        response = requests.get("https://universe.roboflow.com/ds/naJ7QRzCU6?key=6ZPMmnDjm3")
        if response.status_code == 200:
            with open("tiger_dataset.zip", 'wb') as f:
                f.write(response.content)
            print("File downloaded successfully!")
            return True
        else:
            print(f"Failed to download file: HTTP status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def unzip_file():
    import zipfile
    print("Unzipping Started")
    try:
        with zipfile.ZipFile("tiger_dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())
        print("File unzipped successfully!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def write_to_yaml(file_path):
    data = """
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['Tiger']

roboflow:
  workspace: tiger-vliot
  project: tiger-z0d6k
  version: 2
  license: CC BY 4.0
  url: https://universe.roboflow.com/tiger-vliot/tiger-z0d6k/dataset/2
"""
    try:
        with open(file_path, 'w') as yaml_file:
            yaml_file.write(data)
        print(f"Data written to '{file_path}' successfully!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def create_model():
    try:
        print("Model Creation Started")
        execute = "python train.py --img 640 --epochs 10 --data data.yaml --weights yolov5s.pt"
        subprocess.check_call(execute)
        print("Created Model")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    
if __name__ == "__main__":
    folder_name = "Model_Creation_Tiger"  # Specify the name of the folder you want to create
    delete_directory(folder_name)
    create_folder(folder_name)
    navigate_to_folder(folder_name)
    repository_url = "https://github.com/ultralytics/yolov5.git"
    destination_path = "yolov5"  # You can change this to your desired destination path

    if install_gitpython() and clone_repository(repository_url, destination_path):
        navigate_to_folder(destination_path)
        setup_venv()
        install_requirements()
        download_file()
        unzip_file()
        write_to_yaml("data.yaml")
        create_model()
