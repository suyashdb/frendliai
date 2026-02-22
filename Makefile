.PHONY: install run-mock run-gateway run-all client test docker-up docker-down

# --- Local development ---

install:
	pip install -r requirements.txt

run-mock:
	python mock_upstream.py

run-gateway:
	python -m gateway.server

run-all:
	@echo "Starting mock upstream on :9000 and gateway on :8000..."
	python mock_upstream.py & \
	sleep 2 && \
	python -m gateway.server & \
	wait

client:
	python client.py

client-no-summaries:
	python client.py --no-prompt-summary --no-reasoning-summary

# --- Docker ---

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

# --- Testing ---

test:
	python -m pytest tests/ -v

test-integration:
	@echo "Start mock + gateway first (make run-all), then run:"
	@echo "  python client.py"
