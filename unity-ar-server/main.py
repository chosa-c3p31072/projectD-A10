import uvicorn
import os
import io
import random
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Unityから送られてきた画像を保存するフォルダ（確認用）
UPLOAD_DIR = "unity_received_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Unity AR連携用 仮サーバー")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Unityから画像を受け取り、ダミーのスコアを返すエンドポイント
    """
    print(f"通信あり: Unityから {file.filename} を受信しました。")

    # 1. 画像データを読み込む
    image_data = await file.read()
    
    # 2. 画像を保存する（ちゃんと届いているか確認するため）
    # ファイル名が被らないように、適当にリネームしてもOKですが、今回はそのまま保存します
    save_path = f"{UPLOAD_DIR}/{file.filename}"
    try:
        # バイナリデータから画像として開けるかチェック
        image = Image.open(io.BytesIO(image_data))
        # 保存
        image.save(save_path)
        print(f"画像を保存しました: {save_path}")
    except Exception as e:
        print(f"画像の保存に失敗しました（データが壊れている可能性があります）: {e}")
        return JSONResponse(content={"status": "error", "message": "画像データが無効です"}, status_code=400)

    # 3. AI処理の代わり（ダミー処理）
    # ランダムで 0.70 〜 0.99 のスコアを出す
    dummy_score = round(random.uniform(0.7, 0.99), 2)

    # 4. Unityが期待している通りのJSONを返す
    result = {
        "status": "success",
        "score": dummy_score,
        "message": "AI 処理が成功しました (仮)"
    }

    print(f"Unityに返すデータ: {result}")
    print("-" * 30)
    
    return JSONResponse(content=result)

# サーバー起動設定
if __name__ == "__main__":
    # host="0.0.0.0" にすることで、同じWi-Fi内のスマホからアクセス可能になります
    print("--- Unity連携用サーバーを起動します ---")
    print("PCのIPアドレスを確認し、Unityエンジニアに伝えてください。")
    uvicorn.run(app, host="0.0.0.0", port=8000)