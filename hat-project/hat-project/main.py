import uvicorn
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from typing import List, Optional, Dict
import json
import math
import colorsys
import io

# --- 必要なインポート ---
from google.oauth2 import service_account 
from google.cloud import vision 
from PIL import Image

# --- 定数定義 ---
DIM_SHAPE = 10
DIM_IMPRESSION = 2 
DIM_USER = DIM_SHAPE + DIM_IMPRESSION
DIM_SPEC = 10
DIM_DESIGN = 20
DIM_HAT = DIM_SPEC + DIM_DESIGN
DIM_PREFERENCE = DIM_USER + DIM_HAT
DIM_PAIR = DIM_USER + DIM_HAT
BETA = 0.2
LEARNING_RATE = 0.1 # ★★★ 新規 v9: 学習率 (1回の指摘でどれくらい数値を動かすか)

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
    feedbacks = relationship("Feedback", back_populates="user") # v9追加

class Hat(Base):
    __tablename__ = "hats"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    vector_spec = Column(String)
    vector_design = Column(String)
    preferences = relationship("UserPreference", back_populates="hat")
    feedbacks = relationship("Feedback", back_populates="hat") # v9追加

class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    hat_id = Column(Integer, ForeignKey("hats.id"))
    vector_preference = Column(String)
    user = relationship("User", back_populates="preferences")
    hat = relationship("Hat", back_populates="preferences")

# ★★★ 新規追加 v9: フィードバック保存用テーブル
class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    hat_id = Column(Integer, ForeignKey("hats.id"))
    is_satisfied = Column(Boolean) # 満足(True)か不満足(False)か
    user_comment = Column(Text, nullable=True) # 任意コメント
    corrected_design_vector = Column(String, nullable=True) # ユーザーが修正したベクトル
    
    user = relationship("User", back_populates="feedbacks")
    hat = relationship("Hat", back_populates="feedbacks")

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

try:
    credentials = service_account.Credentials.from_service_account_file(KEY_FILE_PATH)
except Exception as e:
    print(f"致命的エラー: サービスアカウントのJSONキーファイルが読み込めません。: {e}")
    credentials = None 

# --- FastAPIアプリケーションの初期化 ---
app = FastAPI(title="【v9 学習機能付き】帽子推薦システム")

# --- 汎用ヘルパー関数 ---
def json_to_vec(json_str: str) -> np.ndarray:
    return np.array(json.loads(json_str))

def vec_to_json(vec: np.ndarray) -> str:
    return json.dumps(vec.tolist())

def get_landmark_pos(landmarks, landmark_type):
    for landmark in landmarks:
        if landmark.type_ == landmark_type: return landmark.position
    return None

def get_distance(p1, p2):
    if p1 is None or p2 is None: return 0
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def diagnose_face_type(v_shape: np.ndarray) -> np.ndarray:
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = v_shape
    score_child_adult = 0.0
    if v1 > 0: score_child_adult += (v1 - 1.5) * 0.5 
    if v7 > 0: score_child_adult -= (v7 - 50.0) * 0.1 
    if v10 > 0: score_child_adult += (v10 - 250.0) * 0.1 
    score_curve_straight = 0.0
    if v9 > 0: score_curve_straight -= (v9 - 10.0) * 0.2 
    if v4 != 0: score_curve_straight += (v4 / 30.0) 
    score_child_adult = np.clip(score_child_adult, -1.0, 1.0)
    score_curve_straight = np.clip(score_curve_straight, -1.0, 1.0)
    return np.array([score_child_adult, score_curve_straight])

def extract_user_vectors(image_bytes: bytes) -> (np.ndarray):
    if not credentials: raise HTTPException(status_code=500, detail="Vision API認証エラー")
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    image = vision.Image(content=image_bytes)
    features = [{"type_": vision.Feature.Type.FACE_DETECTION},]
    response = vision_client.annotate_image(request=vision.AnnotateImageRequest(image=image, features=features))
    if not response.face_annotations: raise HTTPException(status_code=400, detail="顔検出失敗")
    face = response.face_annotations[0]
    landmarks = face.landmarks
    
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
    vector_impression = diagnose_face_type(vector_shape)
    return vector_shape, vector_impression

def extract_color_vector(image_bytes: bytes) -> np.ndarray:
    if not credentials: raise HTTPException(status_code=500, detail="Vision API認証エラー")
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    image = vision.Image(content=image_bytes)
    features = [{"type_": vision.Feature.Type.IMAGE_PROPERTIES},]
    response = vision_client.annotate_image(request=vision.AnnotateImageRequest(image=image, features=features))
    v_h, v_s, v_l = 0.0, 0.0, 0.0
    if response.image_properties_annotation.dominant_colors:
        dom_color = response.image_properties_annotation.dominant_colors.colors[0]
        r, g, b = dom_color.color.red/255.0, dom_color.color.green/255.0, dom_color.color.blue/255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        v_h, v_s, v_l = h, s, l 
    return np.array([v_h, v_s, v_l])

def get_user_vector(user: User) -> np.ndarray:
    return np.concatenate([json_to_vec(user.vector_shape), json_to_vec(user.vector_impression)])

def get_hat_vector(hat: Hat) -> np.ndarray:
    return np.concatenate([json_to_vec(hat.vector_spec), json_to_vec(hat.vector_design)])

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0  
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

# --- API Endpoints ---

@app.post("/users/register/", summary="【利用者用】ユーザー登録")
async def register_user(username: str = Form(...), image: UploadFile = File(...), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == username).first()
    if db_user: raise HTTPException(status_code=400, detail="ユーザー名は既に使用されています")
    image_bytes = await image.read()
    try:
        v_shape, v_impression = extract_user_vectors(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI処理エラー: {e}")
    new_user = User(username=username, vector_shape=vec_to_json(v_shape), vector_impression=vec_to_json(v_impression))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": f"ユーザー '{username}' 登録完了", "user_id": new_user.id, "V_user_impression": v_impression.tolist()}

@app.post("/hats/register/", summary="【管理者用】帽子登録")
async def register_hat(name: str = Form(...), manual_spec_vector_json: str = Form(...), design_vector_json: str = Form(...), image: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        manual_spec = [float(x.strip()) for x in manual_spec_vector_json.split(',')]
        if len(manual_spec) != 7: raise ValueError("測定値は7次元必要")
        v_design = np.array([float(x.strip()) for x in design_vector_json.split(',')])
        if len(v_design) != DIM_DESIGN: raise ValueError(f"デザインは{DIM_DESIGN}次元必要")
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))
    
    image_bytes = await image.read()
    try:
        v_color = extract_color_vector(image_bytes)
    except Exception as e: raise HTTPException(status_code=500, detail=f"AI処理エラー: {e}")
    
    v_spec_final = np.array([manual_spec[0], manual_spec[1], manual_spec[2], manual_spec[3], manual_spec[4], manual_spec[5], v_color[0], v_color[1], v_color[2], manual_spec[6]])
    new_hat = Hat(name=name, vector_spec=vec_to_json(v_spec_final), vector_design=vec_to_json(v_design))
    db.add(new_hat)
    db.commit()
    db.refresh(new_hat)
    return {"message": f"帽子 '{name}' 登録完了", "hat_id": new_hat.id, "V_hat_spec": v_spec_final.tolist()}

@app.post("/users/{user_id}/select-preference/{hat_id}", summary="【利用者用】好みの登録")
async def define_preference_vector(user_id: int, hat_id: int, db: Session = Depends(get_db)):
    user, hat = db.query(User).filter(User.id == user_id).first(), db.query(Hat).filter(Hat.id == hat_id).first()
    if not user or not hat: raise HTTPException(status_code=404, detail="データなし")
    if db.query(UserPreference).filter_by(user_id=user_id, hat_id=hat_id).first(): return {"message": "登録済み"}
    v_pref = np.concatenate([get_user_vector(user), get_hat_vector(hat)])
    db.add(UserPreference(user_id=user_id, hat_id=hat_id, vector_preference=vec_to_json(v_pref)))
    db.commit()
    return {"message": f"好み登録完了: User {user_id} -> Hat {hat_id}"}

@app.get("/recommendations/{user_id}", summary="【利用者用】推薦スコア計算")
async def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user: raise HTTPException(status_code=404, detail="ユーザーなし")
    v_user = get_user_vector(user)
    all_prefs = db.query(UserPreference).all()
    v_avg = np.mean([json_to_vec(p.vector_preference) for p in all_prefs], axis=0) if all_prefs else np.zeros(DIM_PREFERENCE)
    p_pref = db.query(UserPreference).filter_by(user_id=user_id).order_by(UserPreference.id.desc()).first()
    v_p_pref = json_to_vec(p_pref.vector_preference) if p_pref else np.zeros(DIM_PREFERENCE)
    has_p = p_pref is not None
    
    results = []
    for hat in db.query(Hat).all():
        v_pair = np.concatenate([v_user, get_hat_vector(hat)])
        s_p = cosine_similarity(v_p_pref, v_pair) if has_p else 0.0
        s_avg = cosine_similarity(v_avg, v_pair) if all_prefs else 0.0
        s_final = (1.0 - BETA) * s_p + BETA * s_avg if has_p else s_avg
        results.append({"hat_id": hat.id, "hat_name": hat.name, "S_final": s_final})
    return {"recommendations": sorted(results, key=lambda x: x["S_final"], reverse=True)}

# ★★★ 新規機能 v9: フィードバック送信と学習 ★★★
@app.post("/feedback/", summary="【利用者用】フィードバック送信と学習")
async def submit_feedback(
    user_id: int,
    hat_id: int,
    is_satisfied: bool,
    user_comment: Optional[str] = Form(None),
    corrected_design_vector_json: Optional[str] = Form(None, description="修正したい20次元のデザインベクトル(任意)"),
    db: Session = Depends(get_db)
):
    """
    ユーザーが結果に不満足な場合、心理量(デザインベクトル)を修正して送信できる。
    システムはそれを学習し、帽子のデータを更新する。
    """
    hat = db.query(Hat).filter(Hat.id == hat_id).first()
    if not hat: raise HTTPException(status_code=404, detail="帽子が見つかりません")

    # 1. フィードバックを保存
    new_feedback = Feedback(
        user_id=user_id, hat_id=hat_id, is_satisfied=is_satisfied,
        user_comment=user_comment, corrected_design_vector=corrected_design_vector_json
    )
    db.add(new_feedback)

    # 2. 学習プロセス (不満足、かつ修正ベクトルがある場合のみ)
    if not is_satisfied and corrected_design_vector_json:
        try:
            # ユーザーが思う「正しいベクトル」
            correction = np.array([float(x.strip()) for x in corrected_design_vector_json.split(',')])
            if len(correction) != DIM_DESIGN: raise ValueError
            
            # 現在の「帽子のベクトル」
            current_design = json_to_vec(hat.vector_design)

            # ★学習ロジック★: 現在の値を、ユーザーの意見の方向へ少し(LEARNING_RATE分)近づける
            # New = Old + LearningRate * (UserOpinion - Old)
            new_design = current_design + LEARNING_RATE * (correction - current_design)
            
            # 更新して保存
            hat.vector_design = vec_to_json(new_design)
            db.commit()
            return {"message": "フィードバックを受け取り、帽子の心理量データを改善(学習)しました。", "updated_design_vector": new_design.tolist()}
            
        except Exception:
            return {"message": "フィードバックを保存しましたが、ベクトルの形式が不正なため学習は行われませんでした。"}

    db.commit()
    return {"message": "フィードバックを保存しました。ありがとうございます。"}

if __name__ == "__main__":
    print("--- 【v9 学習機能付き】帽子推薦システムAPI を起動します ---")
    print("APIドキュメント: http://127.0.0.1:8081/docs")
    uvicorn.run(app, host="127.0.0.1", port=8081)