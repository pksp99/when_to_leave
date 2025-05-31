DATA_DIR=data
MODEL_DIR=models

clean_all: clean_data clean_models clean_pycache

clean_data:
	rm -f $(DATA_DIR)/*.pkl $(DATA_DIR)/*.csv $(DATA_DIR)/*.json

clean_models:
	rm -f $(MODEL_DIR)/*.pkl $(MODEL_DIR)/*.zip $(MODEL_DIR)/*.json

clean_pycache:
	find . -type d -name "__pycache__" -exec rm -rf {} +