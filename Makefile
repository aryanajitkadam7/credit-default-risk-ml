export PYTHONPATH := src

install:
	pip install -r requirements.txt

train:          ## all experiments + tuning, logs to MLflow, writes reports/results.json
	python -m credit_risk.train --trials 30

explain:        ## SHAP summary plot + top features
	python -m credit_risk.explain

drift:          ## PSI drift report
	python -m credit_risk.drift

test:
	pytest -q

serve:
	uvicorn credit_risk.serve:app --reload --port 8000

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker:
	docker build -t credit-risk-api . && docker run -p 8000:8000 credit-risk-api
