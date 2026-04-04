.PHONY: all setup extract-frames crop-students split-data train-baseline evaluate-baseline generate-telemetry generate-gradcam tsne benchmark visualize-video plot-curves diagram publish-model deploy-space

all: extract-frames crop-students split-data train-baseline evaluate-baseline generate-telemetry generate-gradcam tsne benchmark visualize-video plot-curves

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

tsne:
	python src/eval/visualize_embeddings.py

benchmark:
	python src/models/optimize_onnx.py

visualize-video:
	python src/eval/visualize_video.py

plot-curves:
	python src/eval/plot_curves.py

diagram:
	awsdac src/docs/aws_architecture.yaml -o results/aws_enterprise_architecture.png

publish-model:
	python src/models/publish_to_huggingface.py

deploy-space:
	python src/models/create_hf_space.py
