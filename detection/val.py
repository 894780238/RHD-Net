import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR

if __name__ == '__main__':
    model = YOLO(r'D:\zenghe\yolov12\runs\train\yolov8-RWKV-diff10-abs2-HRASFF8+last2\weights/best_renamed.pt')
    model.val(data=r'D:\zenghe\\yolov12\data_Strip\data_Strip.yaml',
              split='test',
              imgsz=640,
              batch=8,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='RHD-Net+data_Strip_rename'
              # name='exp',
              )