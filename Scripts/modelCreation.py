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
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def download_file():
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
    try:
        with zipfile.ZipFile("tiger_dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())
        print("File unzipped successfully!")
        return True
    except Exception as e:
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
