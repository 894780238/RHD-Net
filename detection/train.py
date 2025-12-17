import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR

if __name__ == '__main__':
    # List of model versions to iterate through
    model_versions = [  'yolov8-RWKV-diff10-abs2-HRASFF8'
                      ]  # Add your desired model versions here

    for version in model_versions:
        print(f"Training with model: {version}")

        # Construct the yaml file name and the run name using the current version
        model_yaml_file = f'{version}.yaml'
        run_name = version

        # Initialize the YOLO model with the specific yaml file
        model = YOLO(model_yaml_file)

        # Optional: Load pre-trained weights.
        # Consider if this is appropriate for your research.
        # if version == 'yolov8s': # Example: only load for a specific version
        #     model.load(f'{version}.pt')

        # Train the model
        model.train(data=r'D:\zenghe\yolov12\weld_seam_strength\weld_seam_strength.yaml',
                    # If your task is different, find 'ultralytics/cfg/default.yaml'
                    # and modify the 'task' (e.g., detect, segment, classify, pose)
                    cache=False,
                    imgsz=640,
                    epochs=200,
                    single_cls=False,  # Set to True for single-class detection
                    batch=8,
                    close_mosaic=0,
                    workers=4,
                    device='0',
                    optimizer='SGD', # Using SGD
                    # seed = 42,
                    # To resume training, set the path to last.pt
                    # resume=f'runs/train/{run_name}_previous_experiment/weights/last.pt',
                    amp=True,  # Turn off if training loss becomes NaN
                    project='runs/train',
                    name=run_name,
                    # resume=True
                    )
        print(f"Finished training for model: {version}")
        print("-" * 30) # Separator for clarity

    print("All training sessions completed.")

    '''
    'yolov12n','yolov12s','yolov12m', 'yolov12l', 'yolov12x','yolov12n','yolo11s','yolo11m', 'yolo11l', 'yolo11x', 'yolov10b', 'yolov10l', 'yolov10m', 'yolov10n', 'yolov10s', 'yolov10x', 'yolov9c', 'yolov9e', 'yolov9m', 'yolov9s',
                          'yolov9t', 'yolov8m', 'yolov8l', 'yolov8x', 'yolov6n', 'yolov6s', 'yolov6m', 'yolov6l', 'yolov6x', 'yolov5n', 'yolov5s', 'yolov5m',
                          'yolov5l', 'yolov5x', 'yolov3'
    '''