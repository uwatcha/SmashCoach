import cv2
from ultralytics import YOLO

# 動画ファイルのパス
video_path = 'videos/converted.mp4'

# Yolov8モデルのロード
model = YOLO('runs/detect/train2/weights/best.pt')

# 動画ファイルの読み込み
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("動画ファイルが開けません。パスやコーデックを確認してください。")
    exit()

# 出力動画の設定
output_path = 'output/detected_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # フレームごとに物体検知を行う
        results = model(frame)
        
        # 検知結果を描画
        annotated_frame = results[0].plot()
        
        # 出力動画にフレームを書き込む
        out.write(annotated_frame)
        
        # フレームを表示
        cv2.imshow('Frame', annotated_frame)
        
        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()
