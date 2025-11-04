import uvicorn
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from typing import List, Optional, Dict
import json
import math
import colorsys
import io

# --- 必要なインポート ---
from google.oauth2 import service_account 

# --- Google AI クライアントのインポート ---
from google.cloud import vision # (安定版)
from PIL import Image

# --- 定数定義 (v8 - 測定値手動入力版) ---
DIM_SHAPE = 10        # V_shape (形状ベクトル)
DIM_IMPRESSION = 2    # V_impression (顔タイプ2次元)
DIM_USER = DIM_SHAPE + DIM_IMPRESSION  # 12次元

# ★★★ 修正 v8 ★★★
DIM_SPEC = 10         # V_spec (測定値ベクトル - 手動7 + 自動3 = 10次元)
DIM_DESIGN = 20       # V_design (デザインベクトル - 変更なし)
DIM_HAT = DIM_SPEC + DIM_DESIGN      # 30次元

DIM_PREFERENCE = DIM_USER + DIM_HAT  # 42次元
DIM_PAIR = DIM_USER + DIM_HAT        # 42次元
BETA = 0.2

# --- データベース設定 ---
DATABASE_URL = "sqlite:///./recommend.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- データベース モデル ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    vector_shape = Column(String)
    vector_impression = Column(String)
    preferences = relationship("UserPreference", back_populates="user")

class Hat(Base):
    __tablename__ = "hats"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    vector_spec = Column(String)
    vector_design = Column(String)
    preferences = relationship("UserPreference", back_populates="hat")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    hat_id = Column(Integer, ForeignKey("hats.id"))
    vector_preference = Column(String)
    user = relationship("User", back_populates="preferences")
    hat = relationship("Hat", back_populates="preferences")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Google API クライアントの初期化 ---

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# !! 重要 !! (あなたのパスをそのまま使っています)
KEY_FILE_PATH = r"/etc/secrets/hat-project-475817-4b0c4fc1e725.json"
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

try:
    credentials = service_account.Credentials.from_service_account_file(KEY_FILE_PATH)
except Exception as e:
    print(f"致命的エラー: サービスアカウントのJSONキーファイルが読み込めません。パスを確認してください。: {e}")
    credentials = None 

# --- FastAPIアプリケーションの初期化 ---
app = FastAPI(title="【v8 測定値手動入力版】帽子推薦システム")


# === ステップ1: 特徴量のベクトル化 (AI実装) ===

# --- 汎用ヘルパー関数 ---
def json_to_vec(json_str: str) -> np.ndarray:
    return np.array(json.loads(json_str))

def vec_to_json(vec: np.ndarray) -> str:
    return json.dumps(vec.tolist())

def get_landmark_pos(landmarks, landmark_type):
    for landmark in landmarks:
        if landmark.type_ == landmark_type:
            return landmark.position
    return None

def get_distance(p1, p2):
    if p1 is None or p2 is None: return 0
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

#顔タイプ診断ロジック
def diagnose_face_type(v_shape: np.ndarray) -> np.ndarray:
    """
    V_shape (10次元) を使って、顔タイプ診断の2軸を計算する簡易ロジック
    """
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = v_shape

    # 軸1: 子供顔 (-1.0) vs 大人顔 (+1.0)
    score_child_adult = 0.0
    if v1 > 0: score_child_adult += (v1 - 1.5) * 0.5 
    if v7 > 0: score_child_adult -= (v7 - 50.0) * 0.1 
    if v10 > 0: score_child_adult += (v10 - 250.0) * 0.1 

    # 軸2: 曲線 (-1.0) vs 直線 (+1.0)
    score_curve_straight = 0.0
    if v9 > 0: score_curve_straight -= (v9 - 10.0) * 0.2 
    if v4 != 0: score_curve_straight += (v4 / 30.0) 

    score_child_adult = np.clip(score_child_adult, -1.0, 1.0)
    score_curve_straight = np.clip(score_curve_straight, -1.0, 1.0)

    print(f"DEBUG: (AI) 顔タイプ診断スコア: [子供/大人: {score_child_adult}, 曲線/直線: {score_curve_straight}]")
    return np.array([score_child_adult, score_curve_straight])


# --- 1a & 1b. V_shape と V_impression を抽出 ---
# (v7から変更なし)
def extract_user_vectors(image_bytes: bytes) -> (np.ndarray):
    """
    (AI実装 v7) V_shape(形状)を抽出し、それを使ってV_impression(顔タイプ)を診断する
    """
    if not credentials:
        raise HTTPException(status_code=500, detail="Vision APIの認証情報がありません(JSONキーファイル)。")
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    
    print("DEBUG: (AI) V_shape 抽出中...")
    image = vision.Image(content=image_bytes)
    
    features = [{"type_": vision.Feature.Type.FACE_DETECTION},]
    request = vision.AnnotateImageRequest(image=image, features=features)
    response = vision_client.annotate_image(request=request)
    
    if not response.face_annotations:
        raise HTTPException(status_code=400, detail="顔が検出されませんでした。")
        
    face = response.face_annotations[0]
    landmarks = face.landmarks

    # V_shape (10次元) の計算
    left_eye = get_landmark_pos(landmarks, vision.FaceAnnotation.Landmark.Type.LEFT_EYE)
    right_eye = get_landmark_pos(landmarks, vision.FaceAnnotation.Landmark.Type.RIGHT_EYE)
    forehead_glabella = get_landmark_pos(landmarks, vision.FaceAnnotation.Landmark.Type.FOREHEAD_GLABELLA)
    chin_gnathion = get_landmark_pos(landmarks, vision.FaceAnnotation.Landmark.Type.CHIN_GNATHION)
    left_cheek = get_landmark_pos(landmarks, vision.FaceAnnotation.Landmark.Type.LEFT_CHEEK_CENTER)
    right_cheek = get_landmark_pos(landmarks, vision.FaceAnnotation.Landmark.Type.RIGHT_CHEEK_CENTER)
    mouth_center = get_landmark_pos(landmarks, vision.FaceAnnotation.Landmark.Type.MOUTH_CENTER)
    nose_tip = get_landmark_pos(landmarks, vision.FaceAnnotation.Landmark.Type.NOSE_TIP)
    upper_lip = get_landmark_pos(landmarks, vision.FaceAnnotation.Landmark.Type.UPPER_LIP)
    lower_lip = get_landmark_pos(landmarks, vision.FaceAnnotation.Landmark.Type.LOWER_LIP)

    face_height = get_distance(forehead_glabella, chin_gnathion)
    face_width = get_distance(left_cheek, right_cheek)
    v1 = (face_height / face_width) if face_width > 0 else 1.0
    v2 = face.pan_angle
    v3 = face.tilt_angle
    v4 = face.roll_angle
    v5 = get_distance(forehead_glabella, left_eye)
    v6 = get_distance(forehead_glabella, left_eye) 
    v7 = get_distance(left_eye, right_eye)
    v8 = get_distance(nose_tip, mouth_center)
    v9 = get_distance(upper_lip, lower_lip)
    v10 = nose_tip.y if nose_tip else 0
    
    vector_shape = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])
    vector_shape = np.nan_to_num(vector_shape, nan=0.0) 
    print("DEBUG: (AI) V_shape 抽出完了")

    # V_impression (2次元) の計算
    vector_impression = diagnose_face_type(vector_shape)
    
    return vector_shape, vector_impression


# --- 1c. V_spec (のうち色だけ): 測定値ベクトル (Vision API) ---
def extract_color_vector(image_bytes: bytes) -> np.ndarray:
    """
    (AI実装 v8) Cloud Vision API で「色 HSL」(3次元) のみを抽出
    """
    if not credentials:
        raise HTTPException(status_code=500, detail="Vision APIの認証情報がありません(JSONキーファイル)。")
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    
    print("DEBUG: (AI) V_spec (Color) 抽出中...")
    image = vision.Image(content=image_bytes)
    
    features = [{"type_": vision.Feature.Type.IMAGE_PROPERTIES},]
    request = vision.AnnotateImageRequest(image=image, features=features)
    response = vision_client.annotate_image(request=request)
    
    v_h, v_s, v_l = 0.0, 0.0, 0.0
    if response.image_properties_annotation.dominant_colors:
        dom_color = response.image_properties_annotation.dominant_colors.colors[0]
        r = dom_color.color.red / 255.0
        g = dom_color.color.green / 255.0
        b = dom_color.color.blue / 255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        v_h, v_s, v_l = h, s, l 
    
    vector_color = np.array([v_h, v_s, v_l])
    print("DEBUG: (AI) V_spec (Color) 抽出完了")
    return vector_color


# --- 総合ベクトル生成 ---
def get_user_vector(user: User) -> np.ndarray:
    v_shape = json_to_vec(user.vector_shape)
    v_impression = json_to_vec(user.vector_impression)
    return np.concatenate([v_shape, v_impression])

def get_hat_vector(hat: Hat) -> np.ndarray:
    v_spec = json_to_vec(hat.vector_spec)
    v_design = json_to_vec(hat.vector_design)
    return np.concatenate([v_spec, v_design])


# === ステップ4: スコア計算 (ヘルパー関数) ===
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0  
    
    return np.dot(v1, v2) / (norm_v1 * norm_v2)


# === APIエンドポイントの実装 ===

@app.post("/users/register/", summary="【利用者用】ユーザー登録とベクトル生成 (AI)")
async def register_user(
    username: str = Form(...), 
    image: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    db_user = db.query(User).filter(User.username == username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="ユーザー名は既に使用されています")
    
    image_bytes = await image.read()

    try:
        v_shape, v_impression = extract_user_vectors(image_bytes)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"予期せぬエラー (register_user): {e}") 
        raise HTTPException(status_code=500, detail=f"AI処理中に予期せぬエラー: {e}")
    
    new_user = User(
        username=username,
        vector_shape=vec_to_json(v_shape),
        vector_impression=vec_to_json(v_impression)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "message": f"ユーザー '{username}' を登録しました。",
        "user_id": new_user.id,
        "V_user_shape": v_shape.tolist(),
        "V_user_impression": v_impression.tolist() # 2次元の顔タイプスコア
    }


@app.post("/hats/register/", summary="【管理者用】帽子登録とベクトル生成 (AI)")
async def register_hat(
    name: str = Form(...),
    # ★★★ 修正 v8 ★★★ (新しい入力項目を追加)
    manual_spec_vector_json: str = Form(..., description="手動測定の7次元 (1-6, 10番目) のカンマ区切り文字列"),
    design_vector_json: str = Form(..., description="手動デザインの20次元のカンマ区切り文字列"),
    image: UploadFile = File(...), # 色抽出のために画像も必要
    db: Session = Depends(get_db)
):
    """
    ステップ1 (帽子): v8
    - 手動測定ベクトル (7次元) を入力
    - 手動デザインベクトル (20次元) を入力
    - 画像から色 (3次元) を抽出
    - これらを結合して帽子ベクトルをDBに保存
    """
    # --- 手動入力のベクトルをパース ---
    try:
        manual_spec_list = [float(x.strip()) for x in manual_spec_vector_json.split(',')]
        if len(manual_spec_list) != 7: # 1-6番目と10番目
            raise ValueError("手動測定ベクトルは7次元である必要があります。")
    except Exception:
        raise HTTPException(status_code=400, detail=f"manual_spec_vector_jsonは7次元のカンマ区切り文字列である必要があります。")

    try:
        v_design_list = [float(x.strip()) for x in design_vector_json.split(',')]
        if len(v_design_list) != DIM_DESIGN:
            raise ValueError
        v_design = np.array(v_design_list)
    except Exception:
        raise HTTPException(status_code=400, detail=f"design_vector_jsonは{DIM_DESIGN}次元のカンマ区切り文字列である必要があります。")

    # --- 画像から色ベクトル (3次元) を抽出 ---
    image_bytes = await image.read()
    try:
        v_color = extract_color_vector(image_bytes) # H, S, L
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"予期せぬエラー (register_hat - 色抽出): {e}") 
        raise HTTPException(status_code=500, detail=f"AI処理中に予期せぬエラー: {e}")
    
    # --- 最終的な V_spec (10次元) を組み立てる ---
    # manual_spec_list は [v1, v2, v3, v4, v5, v6, v10] の順
    # v_color は [v7(H), v8(S), v9(L)] の順
    v_spec_final = np.array([
        manual_spec_list[0], # v1
        manual_spec_list[1], # v2
        manual_spec_list[2], # v3
        manual_spec_list[3], # v4
        manual_spec_list[4], # v5
        manual_spec_list[5], # v6
        v_color[0],          # v7 (H)
        v_color[1],          # v8 (S)
        v_color[2],          # v9 (L)
        manual_spec_list[6]  # v10
    ])

    # --- データベースに保存 ---
    new_hat = Hat(
        name=name,
        vector_spec=vec_to_json(v_spec_final),
        vector_design=vec_to_json(v_design)
    )
    db.add(new_hat)
    db.commit()
    db.refresh(new_hat)
    
    return {
        "message": f"帽子 '{name}' を登録しました。",
        "hat_id": new_hat.id,
        "V_hat_spec": v_spec_final.tolist(),
        "V_hat_design": v_design.tolist()
    }


@app.post("/users/{user_id}/select-preference/{hat_id}", summary="【利用者用】好みベクトルの定義")
async def define_preference_vector(
    user_id: int, 
    hat_id: int, 
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    hat = db.query(Hat).filter(Hat.id == hat_id).first()
    
    if not user or not hat:
        raise HTTPException(status_code=404, detail="ユーザーまたは帽子が見つかりません")
        
    existing_pref = db.query(UserPreference).filter_by(user_id=user_id, hat_id=hat_id).first()
    if existing_pref:
        return {"message": "この好みは既に登録されています。"}

    v_user = get_user_vector(user)
    v_hat = get_hat_vector(hat)
    
    v_preference = np.concatenate([v_user, v_hat])
    
    new_preference = UserPreference(
        user_id=user_id,
        hat_id=hat_id,
        vector_preference=vec_to_json(v_preference)
    )
    db.add(new_preference)
    db.commit()
    
    return {
        "message": f"ユーザーID {user_id} の「好み」として帽子ID {hat_id} を登録しました。",
        "generated_V_preference_dims": len(v_preference)
    }


@app.get("/recommendations/{user_id}", summary="【利用者用】似合い度スコアの自動計算")
async def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
    
    v_user = get_user_vector(user)
    
    all_preferences = db.query(UserPreference).all()
    if not all_preferences:
        # ★★★ 修正 v8 ★★★ (次元数をDIM_PREFERENCEに合わせる)
        v_average_preference = np.zeros(DIM_PREFERENCE) 
        N = 0
    else:
        all_pref_vectors = [json_to_vec(p.vector_preference) for p in all_preferences]
        v_average_preference = np.mean(all_pref_vectors, axis=0)
        N = len(all_pref_vectors)

    personal_pref = db.query(UserPreference).filter_by(user_id=user_id).order_by(UserPreference.id.desc()).first()
    
    if personal_pref:
        v_preference_personal = json_to_vec(personal_pref.vector_preference)
        has_personal_score = True
    else:
        # ★★★ 修正 v8 ★★★ (次元数をDIM_PREFERENCEに合わせる)
        v_preference_personal = np.zeros(DIM_PREFERENCE)
        has_personal_score = False

    all_hats = db.query(Hat).all()
    if not all_hats:
        return {"message": "評価対象の帽子が登録されていません。"}
        
    results = []

    for hat_b in all_hats:
        v_hat_b = get_hat_vector(hat_b)
        v_pair_b = np.concatenate([v_user, v_hat_b])
        
        s_personal = cosine_similarity(v_preference_personal, v_pair_b) if has_personal_score else 0.0
        s_average = cosine_similarity(v_average_preference, v_pair_b) if N > 0 else 0.0

        if not has_personal_score:
            s_final = s_average
        else:
            s_final = (1.0 - BETA) * s_personal + BETA * s_average

        results.append({
            "hat_id": hat_b.id,
            "hat_name": hat_b.name,
            "S_final": s_final,
            "S_personal": s_personal,
            "S_average": s_average
        })

    sorted_results = sorted(results, key=lambda x: x["S_final"], reverse=True)
    
    return {
        "user_id": user_id,
        "username": user.username,
        "average_data_count (N)": N,
        "has_personal_preference": has_personal_score,
        "recommendations": sorted_results
    }


# サーバー実行
if __name__ == "__main__":
    print("--- 【v8 測定値手動入力版】帽子推薦システムAPI を起動します ---")
    print("!! Google APIの認証が設定されていることを確認してください !!")
    print("APIドキュメント: http://127.0.0.1:8081/docs")
    uvicorn.run(app, host="127.0.0.1", port=8081)