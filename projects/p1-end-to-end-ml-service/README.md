# p1 — End-to-end ML service

**After Phase 1 · Draws on [G2](../../books/01-hands-on-ml-geron/notes/ch02.md),
[G19](../../books/01-hands-on-ml-geron/notes/ch19.md)**

Take Géron chapter 2 all the way: a trained model behind an HTTP endpoint, in a
container, with the preprocessing shipped inside the artifact.

## Why this project

Chapter 2 stops at a pickled model. Everything that makes it a *service* — the
serving boundary, versioning, the schema, the monitoring — is where projects
actually fail. Doing it once on a simple tabular model means you already know the
shape when you get to [p6](../p6-production-llm-app/).

## Definition of done

- [ ] A dataset split with **[`hash_split`](../../src/aieng/classic/splits.py)**,
      so adding rows never moves a row between train and test
- [ ] A scikit-learn `Pipeline` containing **every** transformation — imputation,
      scaling, encoding, engineered features
- [ ] Hyperparameters tuned on validation; the test set touched **once**
- [ ] Test score reported with a **confidence interval**, not a point estimate
- [ ] The whole pipeline serialized as one artifact and loaded in a fresh process
- [ ] A `POST /predict` endpoint with a typed request/response schema
- [ ] `GET /health` and a version endpoint reporting the model version
- [ ] Dockerfile; `docker run` gives a working service
- [ ] p50/p95 latency measured under load
- [ ] A README stating the metric, the baseline it beats, and the retraining trigger

## The point of this project

**Preprocessing must live inside the artifact.** If your serving code
re-implements the scaling, it will drift from training and your model will quietly
get worse — training/serving skew, invisible in every metric you have.

That single discipline is why the pipeline goes in the pickle, and it is exactly
the same argument as putting preprocessing layers inside a Keras model
([G13](../../books/01-hands-on-ml-geron/notes/ch13.md)).

## Pitfalls

- **Fitting a scaler or imputer on the full dataset.** The most common leak there
  is. Use `Pipeline` so it is structurally impossible.
- **Exploring before splitting.** Data snooping — you cannot un-see a pattern.
- **`OneHotEncoder` without `handle_unknown="ignore"`.** A category that appears
  only in production crashes the transform.
- **No input validation at the endpoint.** Garbage in, confident prediction out.
- **Returning a bare number.** Return the prediction, the model version, and
  (where meaningful) a probability.
- **No monitoring plan.** Models decay. Géron insists; most projects skip it.

## Stretch

- A `/predict/batch` endpoint, and measure the throughput difference. That is the
  same batching argument that dominates LLM serving
  ([H9](../../books/04-ai-engineering-huyen/notes/ch09.md)).
- Log inputs and predictions, then compute distribution drift weekly.
- Shadow deployment: run v2 alongside v1, compare offline, never serve it.
- A model card: data, metrics, intended use, and known limitations.
- Replace the model with a gradient-boosted tree
  ([G7](../../books/01-hands-on-ml-geron/notes/ch07.md)) and confirm the service
  code did not have to change. That is what the artifact boundary buys you.

## Getting started

```bash
pip install -e ".[classic,serving]"
python -m projects.p1_ml_service.train --data data/housing.csv --out artifacts/
uvicorn projects.p1_ml_service.app:app --reload
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"...": ...}'
```
