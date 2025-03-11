# Force redeploy for styled callback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from transformers import pipeline
import numpy as np
from sklearn.neural_network import MLPRegressor
import uvicorn
import os

app = FastAPI()

# Spotify 憑證
CLIENT_ID = "cac8d76e0ecf4c8cb24bda595e5a539b"
CLIENT_SECRET = "1247fc65794040fc80653c51b2657f66"
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8888/callback")  # 本地測試用，雲端需更新
SCOPE = "user-library-read user-read-recently-played playlist-modify-public playlist-read-collaborative"

sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET,
                        redirect_uri=REDIRECT_URI,
                        scope=SCOPE)

# 超強情感分析模型
sentiment_analyzer = pipeline("sentiment-analysis", model="roberta-large")
token_info = {}

# CSS 樣式
STYLE = """
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #191414; color: #fff; }
        h1 { color: #1DB954; }
        p { font-size: 18px; }
        a, button { color: #fff; background: #1DB954; padding: 10px 20px; text-decoration: none; border-radius: 25px; font-weight: bold; margin: 10px; border: none; cursor: pointer; }
        a:hover, button:hover { background: #1ed760; }
        input { padding: 10px; border-radius: 5px; border: none; margin: 10px; }
    </style>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    auth_url = sp_oauth.get_authorize_url()
    return f"""
    <html>
        <head>{STYLE}</head>
        <body>
            <h1>ASI Spotify 夢幻模組</h1>
            <p>用人工超智能探索你的音樂世界</p>
            <a href="{auth_url}">登入 Spotify</a>
        </body>
    </html>
    """

@app.get("/callback", response_class=HTMLResponse)
async def callback(request: Request):
    global token_info
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="認證失敗")
    try:
        token_info = sp_oauth.get_access_token(code)
        return f"""
        <html>
            <head>{STYLE}</head>
            <body>
                <h1>登入成功</h1>
                <p>歡迎使用 ASI Spotify 夢幻模組！</p>
                <a href="/recommend">獲取超智能推薦</a>
                <a href="/mood">生成心情播放清單</a>
            </body>
        </html>
        """
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"認證錯誤: {str(e)}")

@app.get("/recommend", response_class=HTMLResponse)
async def recommend():
    if not token_info:
        return HTMLResponse(f"<html><head>{STYLE}</head><body><p>請先登入：<a href='/'>點我</a></p></body></html>")
    sp = spotipy.Spotify(auth=token_info["access_token"])
    recent_tracks = sp.current_user_recently_played(limit=50)
    track_ids = [item["track"]["id"] for item in recent_tracks["items"]]
    audio_features = sp.audio_features(track_ids)
    features = np.array([[f["energy"], f["tempo"], f["danceability"], f["valence"]] 
                         for f in audio_features if f])
    
    # 用 MLP 預測相似歌曲
    mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp.fit(features, features)  # 自監督學習
    predicted = mlp.predict(features[:1])[0]
    distances = np.linalg.norm(features - predicted, axis=1)
    recommended_idx = np.argmin(distances)
    recommended_track = sp.track(track_ids[recommended_idx])
    
    return f"""
    <html>
        <head>{STYLE}</head>
        <body>
            <h1>超智能推薦</h1>
            <p>推薦歌曲：{recommended_track["name"]} - {recommended_track["artists"][0]["name"]}</p>
            <a href="/mood">試試心情播放清單</a>
        </body>
    </html>
    """

@app.get("/mood", response_class=HTMLResponse)
async def mood_playlist(mood: str = None):
    if not token_info:
        return HTMLResponse(f"<html><head>{STYLE}</head><body><p>請先登入：<a href='/'>點我</a></p></body></html>")
    
    if not mood:  # 顯示輸入表單
        return f"""
        <html>
            <head>{STYLE}</head>
            <body>
                <h1>心情播放清單</h1>
                <p>告訴我你的心情：</p>
                <form action="/mood" method="get">
                    <input type="text" name="mood" placeholder="例如：我今天很開心">
                    <button type="submit">生成播放清單</button>
                </form>
                <a href="/recommend">回到推薦</a>
            </body>
        </html>
        """
    
    sp = spotipy.Spotify(auth=token_info["access_token"])
    sentiment = sentiment_analyzer(mood)[0]
    mood_score = sentiment["score"] if sentiment["label"] == "POSITIVE" else -sentiment["score"]
    mood_query = "happy energetic" if mood_score > 0 else "sad calm"
    search_results = sp.search(q=mood_query, type="track", limit=10)
    track_uris = [track["uri"] for track in search_results["tracks"]["items"]]
    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(user_id, f"ASI Mood Playlist - {mood_query}", public=True)
    sp.playlist_add_items(playlist["id"], track_uris)
    return f"""
    <html>
        <head>{STYLE}</head>
        <body>
            <h1>心情播放清單</h1>
            <p>根據你的心情「{mood}」，已生成播放清單！</p>
            <a href="{playlist["external_urls"]["spotify"]}" target="_blank">在 Spotify 中查看</a>
            <br>
            <a href="/recommend">回到推薦</a>
        </body>
    </html>
    """

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8888))
    uvicorn.run(app, host="0.0.0.0", port=port)