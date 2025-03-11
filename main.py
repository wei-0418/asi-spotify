from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from transformers import pipeline
import numpy as np
from sklearn.cluster import KMeans
import uvicorn

app = FastAPI()

# Spotify 憑證（使用你提供的）
CLIENT_ID = "cac8d76e0ecf4c8cb24bda595e5a539b"
CLIENT_SECRET = "1247fc65794040fc80653c51b2657f66"
REDIRECT_URI = "http://localhost:8888/callback"
SCOPE = "user-library-read user-read-recently-played playlist-modify-public playlist-read-collaborative"

# Spotify 認證
sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET,
                        redirect_uri=REDIRECT_URI,
                        scope=SCOPE)

# Hugging Face 情感分析模型（模擬 ASI）
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 儲存 token
token_info = {}

# 首頁：登入 Spotify
@app.get("/", response_class=HTMLResponse)
async def index():
    auth_url = sp_oauth.get_authorize_url()
    return f"""
    <html>
        <body>
            <h1>ASI Spotify 夢幻模組</h1>
            <p>歡迎使用人工超智能音樂體驗</p>
            <a href="{auth_url}">登入 Spotify</a>
        </body>
    </html>
    """

# 回呼：處理認證
@app.get("/callback")
async def callback(request: Request):
    global token_info
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="認證失敗")
    token_info = sp_oauth.get_access_token(code)
    return {"message": "登入成功，請前往 /recommend 或 /mood"}

# 超智能推薦
@app.get("/recommend", response_class=HTMLResponse)
async def recommend():
    if not token_info:
        return HTMLResponse("請先登入：<a href='/'>點我</a>")
    
    sp = spotipy.Spotify(auth=token_info["access_token"])
    
    # 抓取最近播放的 50 首歌
    recent_tracks = sp.current_user_recently_played(limit=50)
    track_ids = [item["track"]["id"] for item in recent_tracks["items"]]
    
    # 抓取音頻特徵
    audio_features = sp.audio_features(track_ids)
    features = [[f["energy"], f["tempo"], f["danceability"], f["valence"]] 
                for f in audio_features if f]
    
    # 用 KMeans 聚類（模擬 ASI 的模式識別）
    kmeans = KMeans(n_clusters=5, random_state=42).fit(features)
    labels = kmeans.labels_
    
    # 推薦相似歌曲
    cluster_idx = np.where(labels == labels[0])[0]
    recommended_track_id = track_ids[np.random.choice(cluster_idx)]
    recommended_track = sp.track(recommended_track_id)
    
    return f"""
    <html>
        <body>
            <h1>超智能推薦</h1>
            <p>推薦歌曲：{recommended_track["name"]} - {recommended_track["artists"][0]["name"]}</p>
            <p><a href="{recommended_track["external_urls"]["spotify"]}" target="_blank">在 Spotify 中播放</a></p>
            <a href="/mood">試試心情播放清單</a>
        </body>
    </html>
    """

# 情感分析 + 播放清單生成
@app.get("/mood", response_class=HTMLResponse)
async def mood_playlist(mood: str = "我今天很開心"):
    if not token_info:
        return HTMLResponse("請先登入：<a href='/'>點我</a>")
    
    sp = spotipy.Spotify(auth=token_info["access_token"])
    
    # 使用 Transformers 分析心情
    sentiment = sentiment_analyzer(mood)[0]
    mood_score = sentiment["score"] if sentiment["label"] == "POSITIVE" else -sentiment["score"]
    
    # 根據心情搜尋歌曲
    mood_query = "happy energetic" if mood_score > 0 else "sad calm"
    search_results = sp.search(q=mood_query, type="track", limit=10)
    track_uris = [track["uri"] for track in search_results["tracks"]["items"]]
    
    # 創建播放清單
    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(user_id, f"ASI Mood Playlist - {mood_query}", public=True)
    sp.playlist_add_items(playlist["id"], track_uris)
    
    return f"""
    <html>
        <body>
            <h1>心情播放清單</h1>
            <p>根據你的心情「{mood}」，已生成播放清單！</p>
            <a href="{playlist["external_urls"]["spotify"]}" target="_blank">在 Spotify 中查看</a>
            <br><a href="/recommend">回到推薦</a>
        </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)