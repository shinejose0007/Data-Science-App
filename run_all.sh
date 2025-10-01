#!/bin/bash
set -e
python -m src.train_model
echo "Training finished. Example prediction:"
python -m src.predict_cli --age 30 --years_experience 5 --education 2 --salary 52000
echo "Try RAG demo by running: python -m src.rag"