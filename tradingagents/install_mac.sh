#!/usr/bin/env bash
# TradingAgents 맥북 설치 스크립트
#   사용법: bash install_mac.sh
#   설치 위치를 바꾸려면: TA_DIR=~/원하는/경로 bash install_mac.sh
set -euo pipefail

TA_DIR="${TA_DIR:-$HOME/TradingAgents}"
REPO_URL="https://github.com/TauricResearch/TradingAgents.git"
PY_VERSION="3.13"

info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[주의]\033[0m %s\n' "$1"; }

if [[ "$(uname)" != "Darwin" ]]; then
    warn "macOS가 아닙니다. 계속 진행하지만 바탕화면 실행 아이콘은 macOS용입니다."
fi

# 1. git (없으면 Xcode Command Line Tools 설치 안내)
if ! command -v git >/dev/null 2>&1; then
    info "git이 없습니다. Xcode Command Line Tools 설치 창을 엽니다."
    xcode-select --install || true
    echo "설치가 끝나면 이 스크립트를 다시 실행하세요."
    exit 1
fi

# 2. uv (파이썬 버전 + 가상환경 관리 도구)
if ! command -v uv >/dev/null 2>&1; then
    info "uv 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 3. TradingAgents 내려받기 (이미 있으면 최신으로 업데이트)
if [[ -d "$TA_DIR/.git" ]]; then
    info "기존 설치 업데이트: $TA_DIR"
    git -C "$TA_DIR" pull --ff-only
else
    info "TradingAgents 내려받는 중: $TA_DIR"
    git clone "$REPO_URL" "$TA_DIR"
fi
cd "$TA_DIR"

# 4. 가상환경 + 패키지 설치
info "Python $PY_VERSION 가상환경 생성 및 패키지 설치 (몇 분 걸릴 수 있음)"
[[ -d .venv ]] || uv venv --python "$PY_VERSION"
uv pip install --python .venv/bin/python .

# 5. .env (API 키 파일)
if [[ ! -f .env ]]; then
    cp .env.example .env
fi

set_env() {  # set_env KEY VALUE : .env 의 'KEY=' 줄을 값으로 교체
    local tmp
    tmp="$(mktemp)"
    awk -v k="$1" -v v="$2" 'BEGIN{done=0}
        $0 ~ "^#?"k"=" && !done {print k"="v; done=1; next} {print}
        END{if(!done) print k"="v}' .env > "$tmp"
    mv "$tmp" .env
}

if [[ -t 0 ]]; then
    echo
    echo "사용할 AI(LLM) 서비스의 API 키를 입력하세요 (나중에 $TA_DIR/.env 에서 수정 가능)"
    echo "  1) OpenAI (GPT)   2) Anthropic (Claude)   3) Google (Gemini)   4) 건너뛰기"
    read -r -p "번호 선택 [4]: " choice
    case "${choice:-4}" in
        1) key_name=OPENAI_API_KEY ;;
        2) key_name=ANTHROPIC_API_KEY ;;
        3) key_name=GOOGLE_API_KEY ;;
        *) key_name="" ;;
    esac
    if [[ -n "$key_name" ]]; then
        read -r -s -p "$key_name 값 붙여넣기 (화면에 표시되지 않음): " key_value
        echo
        if [[ -n "$key_value" ]]; then
            set_env "$key_name" "$key_value"
            info "$key_name 저장 완료"
        fi
    fi
fi
chmod 600 .env

# 6. 실행 스크립트 (터미널용 + 더블클릭용)
cat > "$TA_DIR/start.command" <<EOF
#!/usr/bin/env bash
# TradingAgents 실행 (더블클릭 또는 터미널에서 실행)
cd "$TA_DIR"
source .venv/bin/activate
tradingagents "\$@"
EOF
chmod +x "$TA_DIR/start.command"

if [[ -d "$HOME/Desktop" ]]; then
    ln -sf "$TA_DIR/start.command" "$HOME/Desktop/TradingAgents.command"
fi

# 7. 동작 확인
.venv/bin/tradingagents --help >/dev/null
info "설치 완료!"
cat <<EOF

  실행 방법
    - 바탕화면의 'TradingAgents.command' 더블클릭
    - 또는 터미널에서:  $TA_DIR/start.command

  API 키 수정:  open -e $TA_DIR/.env
EOF
