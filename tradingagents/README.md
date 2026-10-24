# TradingAgents 맥북 설치 가이드

[TradingAgents](https://github.com/TauricResearch/TradingAgents)는 여러 AI 에이전트(시장·뉴스·재무 분석가, 강세/약세 연구원, 트레이더, 리스크 관리자)가
토론해서 종목의 매수/매도/보유 의견을 내는 프레임워크입니다. **실제 주문은 넣지 않고 분석 리포트만 만듭니다.**

## 1. 설치 (한 번만)

맥에서 **터미널** 앱을 열고 아래를 붙여넣으세요.

```bash
curl -fsSL https://raw.githubusercontent.com/posthvn/python/claude/sweet-noether-okh7h2/tradingagents/install_mac.sh -o install_mac.sh
bash install_mac.sh
```

이 저장소를 이미 clone 했다면 `bash tradingagents/install_mac.sh` 만 실행하면 됩니다.

스크립트가 하는 일:
1. `git` 확인 (없으면 Xcode Command Line Tools 설치 창을 띄움 → 설치 후 다시 실행)
2. [uv](https://docs.astral.sh/uv/) 설치 (Python 3.13과 가상환경을 알아서 준비)
3. `~/TradingAgents` 에 소스 내려받기 (다시 실행하면 최신 버전으로 업데이트)
4. `~/TradingAgents/.venv` 가상환경에 패키지 설치
5. API 키 입력 받아 `~/TradingAgents/.env` 에 저장
6. 바탕화면에 `TradingAgents.command` 실행 아이콘 생성

## 2. API 키

LLM API 키가 하나 필요합니다 (사용량만큼 과금됩니다).
- OpenAI: https://platform.openai.com/api-keys
- Anthropic(Claude): https://console.anthropic.com/
- Google(Gemini): https://aistudio.google.com/apikey

설치 중 건너뛰었거나 바꾸려면: `open -e ~/TradingAgents/.env` 로 열어 `OPENAI_API_KEY=...` 같은 줄을 채우세요.

## 3. 실행

- 바탕화면 **TradingAgents.command** 더블클릭
  (처음엔 "확인되지 않은 개발자" 경고가 뜰 수 있음 → 아이콘 우클릭 → 열기)
- 또는 터미널에서 `~/TradingAgents/start.command`

화면에서 종목(예: `NVDA`, 한국 주식은 `005930.KS`(삼성전자), 코스닥은 `.KQ`), 분석 날짜, LLM, 분석 깊이를 고르면 됩니다.
출력 언어를 한국어로 고를 수도 있고, `.env` 에 `TRADINGAGENTS_OUTPUT_LANGUAGE=Korean` 을 넣어 고정할 수도 있습니다.

질문 없이 바로 돌리기:

```bash
~/TradingAgents/start.command --ticker 005930.KS --date 2026-10-23 --analysts market,news,fundamentals --save --no-show
```

## 4. Python 코드에서 쓰기

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"   # openai / google / anthropic ...
config["max_debate_rounds"] = 1

ta = TradingAgentsGraph(debug=True, config=config)
state, decision = ta.propagate("NVDA", "2026-10-23")
print(decision)
ta.save_reports(state, "NVDA")  # ~/.tradingagents 아래에 리포트 저장
```

실행할 땐 가상환경의 파이썬을 쓰세요: `~/TradingAgents/.venv/bin/python 내스크립트.py`

> ⚠️ 연구용 프레임워크입니다. 결과는 투자 조언이 아니며, LLM 특성상 같은 입력에도 결과가 달라질 수 있습니다.
