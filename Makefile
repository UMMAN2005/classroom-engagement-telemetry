.PHONY: setup extract-frames crop-students split-data train-baseline evaluate-baseline generate-telemetry generate-gradcam publish-model

setup:
	pip install -r requirements.txt

extract-frames:
	python src/data/extract_frames.py

crop-students:
	python src/data/crop_students.py

split-data:
	python src/data/split_data.py

train-baseline:
	python src/models/train_baseline.py

evaluate-baseline:
	python src/eval/evaluate_baseline.py

generate-telemetry:
	python src/eval/generate_telemetry.py

generate-gradcam:
	python src/eval/generate_gradcam.py

publish-model:
	python src/models/publish_to_huggingface.py
