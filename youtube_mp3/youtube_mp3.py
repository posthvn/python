"""유튜브 동영상에서 오디오만 추출해 MP3로 저장하는 프로그램.

필요 패키지:
    pip install yt-dlp
그리고 시스템에 ffmpeg 가 설치되어 있어야 합니다.
    - Windows: winget install ffmpeg  (또는 https://ffmpeg.org 에서 받아 PATH 등록)
    - macOS:   brew install ffmpeg
    - Ubuntu:  sudo apt install ffmpeg

사용 예:
    python youtube_mp3.py https://www.youtube.com/watch?v=XXXXXXXXXXX
    python youtube_mp3.py URL1 URL2 -o ~/Music -q 320
    python youtube_mp3.py "재생목록URL" --playlist
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError
except ImportError:
    sys.exit("yt-dlp 가 설치되어 있지 않습니다. 먼저 'pip install yt-dlp' 를 실행하세요.")


def build_options(output_dir: Path, quality: str, playlist: bool) -> dict:
    """yt-dlp 옵션 딕셔너리를 만든다."""
    return {
        # 가장 좋은 음질의 오디오 스트림만 받는다.
        "format": "bestaudio/best",
        # 저장 경로/파일명: <출력폴더>/<영상제목>.mp3
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        # ffmpeg 로 MP3 변환 + 메타데이터/썸네일(앨범아트) 삽입
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": quality,
            },
            {"key": "FFmpegMetadata"},
            {"key": "EmbedThumbnail"},
        ],
        "writethumbnail": True,
        "noplaylist": not playlist,
        # 파일명에 쓸 수 없는 문자를 안전하게 바꾼다 (Windows 대응).
        "windowsfilenames": True,
        # 재생목록 중 일부 영상이 실패해도 나머지는 계속 진행.
        "ignoreerrors": playlist,
        "progress_hooks": [progress_hook],
    }


def progress_hook(d: dict) -> None:
    """다운로드 진행 상황을 한 줄로 출력한다."""
    if d["status"] == "downloading":
        percent = d.get("_percent_str", "").strip()
        speed = d.get("_speed_str", "").strip()
        print(f"\r  다운로드 중... {percent}  {speed}   ", end="", flush=True)
    elif d["status"] == "finished":
        print("\r  다운로드 완료, MP3 변환 중...            ")


def download_mp3(urls: list[str], output_dir: Path, quality: str = "192",
                 playlist: bool = False) -> int:
    """URL 목록을 MP3로 저장한다. 실패한 개수를 반환한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    options = build_options(output_dir, quality, playlist)

    failed = 0
    with YoutubeDL(options) as ydl:
        for url in urls:
            print(f"\n▶ {url}")
            try:
                ydl.download([url])
            except DownloadError as e:
                failed += 1
                print(f"  실패: {e}")
    return failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="유튜브 동영상의 소리만 MP3로 저장합니다.")
    parser.add_argument("urls", nargs="*", help="유튜브 동영상 URL (여러 개 가능)")
    parser.add_argument("-o", "--output", default="downloads",
                        help="저장 폴더 (기본값: ./downloads)")
    parser.add_argument("-q", "--quality", default="192",
                        choices=["128", "192", "256", "320"],
                        help="MP3 비트레이트 kbps (기본값: 192)")
    parser.add_argument("-p", "--playlist", action="store_true",
                        help="URL이 재생목록이면 전체를 받는다")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg 를 찾을 수 없습니다. ffmpeg 를 설치하고 PATH 에 등록하세요.")

    urls = args.urls
    if not urls:
        # 인자 없이 실행하면 직접 입력받는다.
        url = input("유튜브 URL을 입력하세요: ").strip()
        if not url:
            sys.exit("URL이 입력되지 않았습니다.")
        urls = [url]

    output_dir = Path(args.output).expanduser().resolve()
    failed = download_mp3(urls, output_dir, args.quality, args.playlist)

    print(f"\n완료! 저장 위치: {output_dir}")
    if failed:
        print(f"실패한 항목: {failed}개")
        sys.exit(1)


if __name__ == "__main__":
    main()
