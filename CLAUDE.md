# TASS 3D Motion Analysis Project

## 개요
Sanders, J.V. (1982) "A three-dimensional dynamic analysis of a towed system",
Ocean Engineering, Vol. 9, No. 5, pp. 483-499 논문을 기반으로 한
3차원 TASS(Towed Array Sonar System) 운동 해석 코드.

## TASS 케이블 4개 섹션 구성

| 섹션 | 이름 | 길이 | 부력 | ds (요소 길이) | 요소 수 | 직경 | 선밀도 | EA |
|------|------|------|------|---------------|---------|------|--------|------|
| 1 | HWC (Heavy Weight Cable) | 1000m | 음성부력 (negative) | 100m | 10 | 35mm | 2.8 kg/m | 3.0e4 N |
| 2 | LWC (Light Weight Cable) | 300m | 중성부력 (neutral) | 50m | 6 | 45mm | 1.63 kg/m | 2.0e4 N |
| 3 | AM (Acoustic Module) | 100m | 중성부력 (neutral) | 20m | 5 | 70mm | 3.95 kg/m | 1.5e4 N |
| 4 | TR (Tail Rope) | 80m | 중성부력 (neutral) | 20m | 4 | 30mm | 0.72 kg/m | 1.0e4 N |

- 전체 길이: 1480m, 전체 요소: 25개, 전체 노드: 26개
- 배치 순서: Tow Point → HWC → LWC → AM → TR (자유단)
- 각 섹션별 FDM 요소 길이(ds)를 독립적으로 설정 가능

## Sanders(1982) 기반 물리 모델

### 지배방정식 (Quasi-Static)
- 접선방향: dT/ds = f_t + w·sin(φ)
- 법선방향(수직): T·dφ/ds = f_n,v - w·cos(φ)
- 법선방향(수평): T·cos(φ)·dθ/ds = f_n,h

### 수치 해법
- **Quasi-static**: 자유단(tail)에서 예인점(tow point)으로 RK4 공간 적분
- **Dynamic**: Lumped-parameter (집중질량) 모델 + RK4 시간 적분
- **수력학적 항력**: Morison 방정식 (접선 마찰 + 법선 압력 항력)
  - 접선: F_t = 0.5 · ρ · Cd_t · π · d · |V_t| · V_t · ds
  - 법선: F_n = 0.5 · ρ · Cd_n · d · |V_n| · V_n · ds

### 수치 안정성 주의사항
- EA 값이 너무 크면 (예: 1e7) explicit RK4에서 dt=0.05와 함께 발산함
- 현재 EA: 1e4 ~ 3e4 범위, 구조 감쇠(damping): 50~120 N·s/m
- Quasi-static solver에서 dφ/ds를 ±0.05 rad/m로 클램핑하여 각도 발산 방지

## 좌표계 규약

- **원점**: 예인점(Tow Point) = (0, 0, 0)
- **X축**: 전진 방향 (예인 방향)
- **Y축**: 횡방향 (우현 양수)
- **Z축**: 심도 방향 (**아래 방향이 양수**, Z+ = depth)
- 시각화에서 Z축은 위쪽이 깊은 쪽으로 표시 (invert_yaxis)

## 파일 구조

| 파일 | 설명 |
|------|------|
| `tass_cable_model.py` | 핵심 모듈 — CableSection, MultiSectionCable, TASSCableModel, TowShipTrajectory, TASSSimulation 클래스 |
| `run_tass_simulation.py` | 시뮬레이션 실행 스크립트 — 4가지 시나리오 (quasi-static, 직진, 선회, S-turn) |
| `tass_visualization.py` | 시각화 모듈 — 3D/2D 플롯, 장력 분포, 시간 이력 |
| `report_tass_analysis.md` | 기술 보고서 (한국어) |
| `results/` | 시뮬레이션 결과 이미지 (.png) |

## 실행 방법

```bash
python run_tass_simulation.py
```

결과 이미지는 `results/` 폴더에 저장됨:
- `qs_steady_state.png` — 정준정적 정상상태
- `straight_*.png` — 직진 예인
- `turn_*.png` — 정상 선회
- `sturn_*.png` — S-turn 기동

## 시뮬레이션 시나리오

1. **Quasi-Static 정상상태**: 예인속도 5 m/s, 정적 평형 형상
2. **직진 예인** (120초): 동적 응답 수렴 확인
3. **정상 선회** (200초): 선회반경 800m
4. **S-Turn** (300초): 선회반경 600m, 위상 지속시간 60초

## Git 브랜치

- 개발 브랜치: `claude/tass-motion-analysis-oKYL8`
