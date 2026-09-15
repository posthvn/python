# 선박 내항성능(Seakeeping) 시뮬레이션

해상상태 5에서 운항하는 선박(L=100 m, 10,000 ton, 5 kts)의 운동을
주파수영역과 시간영역에서 계산하는 파이썬 코드입니다.

## 실행 방법

```bash
pip install numpy matplotlib
python ship_motion_simulation.py
```

실행하면 콘솔에 주파수영역 통계(RMS, 유의 단진폭, 평균 zero-crossing 주기 Tz)가
출력되고, `results/` 폴더에 그림 3장이 저장됩니다.

| 파일 | 내용 |
|---|---|
| `results/frequency_domain.png` | P-M 파랑 스펙트럼, heave/pitch/roll RAO, 응답 스펙트럼 |
| `results/time_series_chi00.png` | 입사각 0° (head sea) 파면·heave·pitch·roll 시계열 |
| `results/time_series_chi30.png` | 입사각 30° (bow sea) 시계열 |

## 계산 조건

| 항목 | 값 | 비고 |
|---|---|---|
| 길이 L | 100 m | 주어짐 |
| 배수량 | 10,000 ton (∇ ≈ 9,756 m³) | 주어짐 |
| 선속 | 5 kts (Fn = 0.082) | 주어짐 |
| 해상상태 | Sea State 5 → Hs = 3.25 m | ITTC/NATO 기준 중간값 |
| 파랑 스펙트럼 | Pierson–Moskowitz 1-파라미터 | 등가풍속 U₁₉.₅ = 12.3 m/s |
| 입사각 χ | 0° (head sea), 30° | 사용자 규약: 0° = 선수파 |
| 폭 B | 16 m | **가정** |
| 방형계수 Cb | 0.70 | **가정** |
| 흘수 T | 8.71 m | ∇ = Cb·L·B·T 로 역산 |
| GM | 1.5 m | **가정** (roll 계산용) |
| Roll 고유주기 | ≈ 10.5 s (kxx = 0.4B) | **가정** |
| Roll 감쇠비 | 0.08 (임계감쇠 대비) | **가정** (빌지킬 포함 수준) |

## 이론 요약

### 파랑 스펙트럼 — P-M 1차(1-파라미터) 스펙트럼

S(ω) = α g²/ω⁵ · exp(−β (ω₀/ω)⁴),  α = 8.1×10⁻³, β = 0.74, ω₀ = g/U₁₉.₅

완전발달해 관계식 Hs = 0.21 U²/g 로부터 Sea State 5 (Hs = 3.25 m)에 해당하는
풍속 U₁₉.₅ ≈ 12.3 m/s (약 24 kts)를 역산하여 사용했습니다.
적분 결과 Hs ≈ 3.23 m, Tp ≈ 9.0 s 로 목표값과 일치합니다.

### RAO — 실험 데이터 대체

요청하신 "입사각별 유사 실험 데이터"의 원본(수치 테이블)은 인터넷에서 직접
내려받을 수 있는 형태로 확보하지 못했습니다(대부분 논문 그림으로만 존재하고,
본 실행 환경에서는 다수 학술 사이트 접근이 차단됨). 대신 **모형시험 및 선형
스트립이론과 비교·검증되어 널리 쓰이는** 반해석적(closed-form) RAO 를
사용했습니다:

> J.J. Jensen, A.E. Mansour, A.S. Olsen,
> *"Estimation of ship motions using closed-form expressions"*,
> Ocean Engineering 31 (2004) 61–85.

heave/pitch(상자형 선체 근사, 비연성 스트립이론):

- 조우주파수: ωₑ = ω + kU·cosχ (k = ω²/g, 심해)
- 유효파수: kₑ = k·|cosχ|, Smith 수정: κ = exp(−kₑT)
- 단면 감쇠: A = 2 sin(ωₑ²B/2g)·exp(−ωₑ²T/g), α = ωₑ/ω
- 기진력: f = √[(1−kT)² + (A²/(kBα³))²]
- 배율: η = 1/√[(1−2kTα²)² + (A²/(kBα²))²]
- X = kₑL/2 일 때
  - **Heave RAO** = η·κ·f·sin(X)/X
  - **Pitch RAO** = η·κ·f·(24/(L(kₑL)²))·(sinX − X·cosX)

극한 검증: 장파장(ω→0)에서 heave→1, pitch→파경사 성분 kₑ 로 수렴함을
확인했습니다. heave·pitch 기진력 간 90° 위상차도 시계열 재현에 반영했습니다.

roll(1자유도, 유효 파경사 기진 모델 — IMO 기상기준 등에서 쓰는 표준형):

φ̈ + 2ζωₙφ̇ + ωₙ²φ = ωₙ²·r·k·e^(−kT)·sinχ·a·cos(ωₑt)

head sea(χ=0)에서는 roll 기진이 0이며, χ=30°에서 roll 이 발생합니다.

### 시간영역 재현

응답 스펙트럼을 조우주파수로 사상한 뒤, 800개 성분의 선형 중첩
(랜덤 위상법)으로 30분 길이의 시계열을 생성합니다:

- ζ(t) = Σ aᵢ·cos(ωₑᵢt + εᵢ), aᵢ = √(2S(ωᵢ)Δω)
- x(t) = Σ aᵢ·|H(ωᵢ)|·cos(ωₑᵢt + εᵢ + arg H(ωᵢ))

시계열 RMS 가 스펙트럼 RMS(√m₀)와 일치함을 콘솔 출력으로 검증합니다.

## 대표 결과 (Sea State 5, 5 kts)

| 응답 | χ=0° 유의 단진폭 | χ=30° 유의 단진폭 |
|---|---|---|
| Heave | 0.61 m | 0.78 m |
| Pitch | 1.9° | 2.0° |
| Roll | 0 | 1.8° |

## 한계 및 빠진 정보 (사용자 확인 필요)

1. **선형(B), 흘수(T), Cb, GM, roll 주기·감쇠는 가정값**입니다. 실제 선박
   값을 `Ship` 클래스에 입력하면 그대로 반영됩니다.
2. **실험 RAO 미사용**: 보유하신 실험 데이터를 `(ω, |H|, 위상)` 테이블로
   주시면 closed-form RAO 를 대체(보간)하도록 쉽게 수정할 수 있습니다.
3. 6자유도 중 **heave, pitch, roll 만** 계산합니다(surge/sway/yaw 제외).
   수직운동 지배 조건(선수파~선수사파)에서는 일반적으로 충분합니다.
4. **장파정(long-crested) 파** 가정 — 방향 분산(spreading) 미포함.
5. 심해 가정, 선형이론(대진폭 비선형 효과·파라메트릭 롤 미포함).
6. heave–pitch 연성, pitch–roll 연성 무시(Jensen 모델의 기본 가정).
