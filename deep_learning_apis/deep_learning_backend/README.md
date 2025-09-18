# Backend Documentation

## Table of Contents
- [Docker Setup](#docker-setup)
  - [Installing Docker](#installing-docker)
  - [Setting Up a Development Environment](#setting-up-a-development-environment)
  - [Setting Up a Production Environment](#setting-up-a-production-environment)
- [Backend Setup and Configuration](#backend-setup-and-configuration)
  - [Python and Environment Management](#python-and-environment-management)
  - [Installing Dependencies with Poetry](#installing-dependencies-with-poetry)
  - [Running the Backend](#running-the-backend)

---

## Docker Setup

### Installing Docker
To use the containerized setup, you must have Docker installed. Follow the [official installation guide](https://www.docker.com/) for your platform.

### Setting Up a Development Environment

0. **Static IP Configuration**:
   - Update the `.env` file with the static IP address for container communication with postgresql database `172.16.1.2` in the "DATABASE_HOST" and "DATABASE_URL".
   - To use a static IP for containers, create a custom Docker network before starting the containers:
     ```bash
     docker network create --gateway 172.16.1.1 --subnet 172.16.1.0/24 deep_learning_backend_subnet
     ```

1. **Create a Directory for Persistent Database Storage**:
   - Navigate to the parent directory of the project root and create a folder named `postgres-data/`:
     ```bash
     mkdir -p ../postgres-data
     ```

2. **Start Containers**:
   - Before execute these comands, open the docker app. Then, Navigate to the `docker-containers/compose` directory, which contains a `docker-compose.yaml` file describing the PostgreSQL and pgAdmin services:
     ```bash
     cd docker-containers/compose/
     docker compose up -d
     ```

3. **Access pgAdmin**:
   - Open your browser and navigate to:
     ```
     http://localhost:5437
     ```
   - Default credentials:
     - **Username:** `example@example.com`
     - **Password:** `87654321`

4. **Connect pgAdmin to PostgreSQL**:
   - Right-click on **Servers** in the left panel, then click:
     ```
     Register -> Server
     ```
   - Use the following settings:
    - In the "General" section:
      - **Name:** `deep_learning` (or any name of your choice).
    - In the "Connection" section:
      - **Host name/address:** Use the IP of the container (to know it, just put on the terminal `docker inspect nome_do_container`. If you are in doubt about the container's name, it is the one with "postgresql" in the name).
      - **Port:** `5432`
      - **Maintenance database:** `postgres`
      - **Username:** `postgres`
      - **Password:** `12345678`

5. **Verify and Create Schema**:
   - The `deeplearning` schema is created automatically by an SQL script loaded during container setup. If needed, you can create it manually in pgAdmin:
     - Navigate to `deep_learning -> Databases -> postgres -> Schemas`.
     - A schema called `deeplearning` will be there.
   
6. **Run Migrations**:
   - In the backend project directory, apply the database migrations after configuring the backend as outlined in [Backend Setup and Configuration](#backend-setup-and-configuration).

   - Also, prepare the environment. Follow the instructions in : [Setting Up a Production Environment](#setting-up-a-production-environment)
   - Then, execute the follow command:
     ```bash
     alembic upgrade head
     ```
   - Finally, execute the backend following the intructions in : [Running the Backend](#running-the-backend)



---

## Backend Setup and Configuration

### Python and Environment Management
1. **Install `pyenv`**:
   - Use [`pyenv`](https://github.com/pyenv/pyenv) for managing Python versions.
   - Update available versions, for **Windows**:
     ```bash
     pyenv update
     ```
   - for **Mac** : 
     ```bash
     brew upgrade pyenv
     ```
   
   - Install necessary system dependencies:
     ```bash
     sudo apt install libssl-dev libffi-dev libncurses5-dev zlib1g zlib1g-dev libreadline-dev libbz2-dev libsqlite3-dev liblzma-dev make gcc
     ```
     
   - Install Python 3.12:
     ```bash
     pyenv install 3.12:latest
     pyenv install 3.12.7
     pyenv global 3.12.7  # Set global version
     pyenv local 3.12.7   # Set local version for the project
     ```

2. **Install `pipx`**:
   - `pipx` allows isolated installation of Python packages:
     ```bash
     pip install pipx
     pipx ensurepath
     exec "$SHELL"
     ```

3. **Install `poetry`**:
   - Use `pipx` to install Poetry:
     ```bash
     pipx install poetry
     exec "$SHELL"
     ```

### Installing Dependencies with Poetry
1. **Install Dependencies**:
   - Navigate to the project root directory and run:
     ```bash
     poetry install
     ```

2. **Activate the Virtual Environment**:
   ```bash
   poetry shell
   ```

3. **Manage Environments**:
   - To remove a virtual environment:
     ```bash
     poetry env remove [ENV_NAME]
     ```
   - To list all environments:
     ```bash
     poetry env list
     ```

---
## Running the Backend
1. **Start the Backend (port 8000)**:
   ```bash
   task run
   ```

1. **Start the Backend (port 8080)**:
   ```bash
   task uvicorn_main
   ```

2. **Run Tests**:
   ```bash
   task test
   ```

3. **Format Code**:
   ```bash
   task format

---
## How to run after all the installations?
1. Update the Database
  ```bash
   alembic upgrade head 
  ```

2. Run the container
  ```bash
   start container 
  ```
3. Start the backend
  ```bash
   task run
  ```
