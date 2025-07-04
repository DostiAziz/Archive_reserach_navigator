install:
	@echo "Installing dependencies..."
	pip install upgrade pip
	pip install -r requirements.txt

lint:
	@echo "Running linters..."
	pylint --disable-R,C src/*.py

format:
	@echo "Formatting code..."
	black src/*.py

#test:
#	@echo "Running tests..."
#	pytest -m pytest tests/ -v
#
#clean:
#	@echo "Cleaning up..."
#	find . -type f -name "*.pyc" -delete
#	find . -type d -name "__pycache__" -delete

run:
	@echo "Running the application..."
	streamlit run src/Main.py

docker-build:
	@echo "Building Docker image..."
	docker compose build

docker-up:
	@echo "Starting Docker containers..."
	docker compose up -d

docker-down:
	@echo "Stopping Docker containers..."
	docker compose down

setup: install
	@echo "Setting up the environment..."
	@echo "Run 'make docker-build' to build the Docker image."
	@echo "Run 'make docker-up' to start the Docker containers."
	@echo "Run 'make docker-down' to stop the Docker containers."