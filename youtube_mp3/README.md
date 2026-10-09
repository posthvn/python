# YouTube → MP3

유튜브 동영상에서 소리만 추출해 MP3로 저장합니다.

## 준비
```bash
pip install -r requirements.txt   # yt-dlp
```
ffmpeg 도 설치되어 있어야 합니다 (Windows: `winget install ffmpeg`, macOS: `brew install ffmpeg`, Ubuntu: `sudo apt install ffmpeg`).

## 사용법
```bash
python youtube_mp3.py                                   # 실행 후 URL 입력
python youtube_mp3.py https://youtu.be/XXXXXXXXXXX
python youtube_mp3.py URL1 URL2 -o ~/Music -q 320       # 여러 개, 저장 폴더/음질 지정
python youtube_mp3.py "재생목록URL" --playlist           # 재생목록 전체
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `-o`, `--output` | 저장 폴더 | `./downloads` |
| `-q`, `--quality` | 비트레이트(kbps): 128/192/256/320 | 192 |
| `-p`, `--playlist` | 재생목록 전체 다운로드 | 끔 |

> 본인이 권리를 가졌거나 다운로드가 허용된 콘텐츠에만 사용하세요.
