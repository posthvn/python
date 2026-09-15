# -*- coding: utf-8 -*-
"""
선박 내항성능(Seakeeping) 시뮬레이션
=====================================

조건
----
1. 선박: 길이 L = 100 m, 배수량 10,000 ton, 선속 5 knots
2. 해상상태(Sea State) 5  →  유의파고 Hs ≈ 3.25 m (ITTC/NATO 기준 중간값)
3. 파랑 스펙트럼: Pierson-Moskowitz 1-파라미터(풍속) 스펙트럼
4. 입사각(χ): 0° (head sea, 선수파) 및 30°
5. RAO(주파수응답함수): 실험 데이터 대신, 모형시험과 비교·검증된
   Jensen, Mansour & Olsen (2004) "Estimation of ship motions using
   closed-form expressions", Ocean Engineering 31, 61-85 의
   반해석적(closed-form) RAO 를 사용 (상세는 README 참조)

출력
----
- frequency domain: 파랑 스펙트럼, RAO, 응답 스펙트럼, 통계치(RMS, 유의진폭, Tz)
- time domain    : 파면(wave elevation), heave, pitch, roll 실시간 시계열

작성: Claude Code 세션 (posthvn 요청)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 화면 없이 그림 파일로 저장
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=20260915)  # 재현 가능한 난수(위상)

G = 9.80665  # 중력가속도 [m/s^2]
RHO = 1025.0  # 해수 밀도 [kg/m^3]

# ---------------------------------------------------------------------------
# 1. 선박 주요목 (주어진 값 + 가정값)
# ---------------------------------------------------------------------------
# 주어진 것: L, 배수량, 선속. 나머지(B, T, Cb, GM 등)는 100 m/10,000 ton 급
# 일반 화물선 기준의 '가정값'이며, 실제 선박 값으로 바꾸면 바로 반영됩니다.
class Ship:
    L = 100.0                      # 수선간장 [m]           (주어짐)
    disp_ton = 10_000.0            # 배수량 [ton]           (주어짐)
    U = 5.0 * 0.514444             # 선속 5 kts -> [m/s]    (주어짐)

    Cb = 0.70                      # 방형계수               (가정)
    B = 16.0                       # 폭 [m]                 (가정)
    # 흘수는 배수량과 일관되도록 역산: disp_vol = Cb * L * B * T
    disp_vol = disp_ton * 1000.0 / RHO          # 배수용적 [m^3] ≈ 9756
    T = disp_vol / (Cb * L * B)                 # 흘수 [m] ≈ 8.7

    GM = 1.5                       # 횡메타센터 높이 [m]    (가정, roll용)
    roll_damp = 0.08               # roll 감쇠비(임계감쇠 대비, 빌지킬 포함 가정)
    kappa_phi = 0.40               # roll 회전반경/폭 비 (kxx = 0.40*B 가정)
    r_eff = 0.85                   # 유효 파경사 계수 (effective wave slope)

    @classmethod
    def Fn(cls):
        return cls.U / np.sqrt(G * cls.L)

    @classmethod
    def roll_natural_freq(cls):
        # T_roll = 2*pi*kxx / sqrt(g*GM)  (관성반경 기반 표준식)
        kxx = cls.kappa_phi * cls.B
        return np.sqrt(G * cls.GM) / kxx        # [rad/s]


# ---------------------------------------------------------------------------
# 2. Pierson-Moskowitz 1-파라미터 스펙트럼
# ---------------------------------------------------------------------------
# S(w) = alpha * g^2 / w^5 * exp( -beta * (w0/w)^4 ),  w0 = g / U19.5
# 완전발달해에서 Hs = 0.21 * U19.5^2 / g  이므로,
# Sea State 5 의 Hs = 3.25 m 에 해당하는 풍속 U19.5 를 역산해 사용한다.
PM_ALPHA = 8.1e-3
PM_BETA = 0.74


def pm_wind_speed_from_Hs(Hs):
    """유의파고 Hs[m] -> 완전발달 P-M 등가 풍속 U19.5 [m/s]"""
    return np.sqrt(Hs * G / 0.21)


def pm_spectrum(w, U_wind):
    """P-M 1-파라미터 스펙트럼 S(w) [m^2 s], w [rad/s]"""
    w = np.asarray(w, dtype=float)
    w0 = G / U_wind
    S = np.zeros_like(w)
    m = w > 1e-6
    S[m] = PM_ALPHA * G**2 / w[m] ** 5 * np.exp(-PM_BETA * (w0 / w[m]) ** 4)
    return S


# ---------------------------------------------------------------------------
# 3. RAO — Jensen, Mansour & Olsen (2004) closed-form
# ---------------------------------------------------------------------------
# 가정: 상자형(box) 선체 근사, heave-pitch 비연성, 선형 스트립이론.
#  - 입사각 규약: chi = 0° 가 head sea (사용자 규약).
#    Jensen 논문의 beta(=180°가 head sea)와는 cos(beta) = -cos(chi) 관계.
#  - 조우주파수  we = w + k*U*cos(chi),  k = w^2/g (심해)
#  - 유효파수    ke = k*|cos(chi)|
#  - Smith 수정  kappa = exp(-ke*T)
#  - 단면 감쇠비 A = 2*sin(we^2*B/(2g)) * exp(-we^2*T/g)
#  - alpha = we/w = 1 + (w*U/g)*cos(chi)
#  - 강제항     f = sqrt( (1-k*T)^2 + (A^2/(k*B*alpha^3))^2 )
#  - 응답배율   eta = 1 / sqrt( (1-2*k*T*alpha^2)^2 + (A^2/(k*B*alpha^2))^2 )
#  - X = ke*L/2 로 두면
#      heave RAO  = eta * kappa * f * sin(X)/X
#      pitch RAO  = eta * kappa * f * (24/(L*(ke*L)^2)) * (sin X - X cos X)
#    (긴 파장 극한에서 heave→1, pitch→파경사 성분 ke 로 수렴함을 확인)
#  - heave 와 pitch 의 기진력은 90° 위상차 (cos vs sin 적분) — 시계열 재현 시 반영.


def encounter_frequency(w, chi_deg, U):
    """조우주파수 we [rad/s] (chi=0: head sea, 양수 방향으로 증가)"""
    k = w**2 / G
    return w + k * U * np.cos(np.radians(chi_deg))


def rao_heave_pitch(w, chi_deg, ship=Ship):
    """
    복소 RAO 반환 (위상 포함).
      H_heave : [m/m],  H_pitch : [rad/m]
    """
    w = np.asarray(w, dtype=float)
    chi = np.radians(chi_deg)
    L, B, T, U = ship.L, ship.B, ship.T, ship.U

    k = w**2 / G                          # 심해 파수
    alpha = 1.0 + (w * U / G) * np.cos(chi)   # we/w
    we = alpha * w                        # 조우주파수
    ke = np.maximum(k * abs(np.cos(chi)), 1e-9)  # 유효파수(0 나눗셈 방지)

    kappa = np.exp(-ke * T)               # Smith 수정
    A = 2.0 * np.sin(we**2 * B / (2 * G)) * np.exp(-(we**2) * T / G)

    kB = np.maximum(k * B, 1e-12)
    # 기진력(복소): Froude-Krylov 성분 + 회절(속도 위상, +i) 성분
    f_c = (1.0 - k * T) + 1j * (A**2 / (kB * alpha**3))
    # 운동방정식 분모(복소): 복원 - 관성 + i*감쇠
    D = (1.0 - 2.0 * k * T * alpha**2) + 1j * (A**2 / (kB * alpha**2))

    X = ke * L / 2.0
    sincX = np.sinc(X / np.pi)            # sin(X)/X  (X→0 에서 1)
    # (sinX - X cosX)/X^3 -> 1/3 (X→0)
    g_geom = np.where(X > 1e-6, (np.sin(X) - X * np.cos(X)) / X**3, 1.0 / 3.0)

    H_heave = (f_c / D) * kappa * sincX
    # pitch: 24/(L*(keL)^2)*(sinX - XcosX) = (6/L)*X*g_geom,
    # 기진 모멘트는 heave 대비 90° 위상차(-i), 파 진행방향 부호 반영
    sgn = np.sign(np.cos(chi)) if abs(np.cos(chi)) > 1e-9 else 1.0
    H_pitch = -1j * sgn * (f_c / D) * kappa * (6.0 / L) * X * g_geom
    return H_heave, H_pitch


def rao_roll(w, chi_deg, ship=Ship):
    """
    Roll 복소 RAO [rad/m] — 1자유도 모델 (유효 파경사 기진).
      phi'' + 2*z*wn*phi' + wn^2*phi = wn^2 * r_eff * k*sin(chi) * a * e^{i we t}
    head sea(chi=0)에서는 roll 기진이 0.
    """
    w = np.asarray(w, dtype=float)
    chi = np.radians(chi_deg)
    k = w**2 / G
    we = encounter_frequency(w, chi_deg, ship.U)
    wn = ship.roll_natural_freq()
    z = ship.roll_damp
    # 파경사의 횡방향 성분 (단파장에서의 압력 감쇠 exp(-k*T) 포함)
    exc = ship.r_eff * k * np.exp(-k * ship.T) * abs(np.sin(chi))
    H = wn**2 * exc / (wn**2 - we**2 + 2j * z * wn * we)
    return H


# ---------------------------------------------------------------------------
# 4. 주파수영역 해석: 응답 스펙트럼과 통계치
# ---------------------------------------------------------------------------
def response_statistics(w, S_wave, H):
    """
    응답 스펙트럼 S_r(w) = |H|^2 * S(w) 와 통계치.
    시간영역 통계(Tz 등)는 조우주파수 가중 모멘트로 계산.
    """
    Sr = np.abs(H) ** 2 * S_wave
    m0 = np.trapezoid(Sr, w)
    return Sr, m0


def spectral_moments_encounter(w, we, Sr):
    """조우주파수 기준 모멘트 m0, m2 -> Tz = 2*pi*sqrt(m0/m2)"""
    m0 = np.trapezoid(Sr, w)
    m2 = np.trapezoid(we**2 * Sr, w)
    return m0, m2


# ---------------------------------------------------------------------------
# 5. 시간영역 재현 (선형 중첩, 랜덤 위상)
# ---------------------------------------------------------------------------
def make_time_series(w, S_wave, chi_deg, t, ship=Ship):
    """
    파면과 3자유도(heave, pitch, roll) 시계열 생성.
      zeta(t)  = sum a_i cos(we_i t + eps_i)
      x(t)     = sum a_i |H_i| cos(we_i t + eps_i + arg H_i)
    """
    dw = np.gradient(w)
    amp = np.sqrt(2.0 * S_wave * dw)          # 각 성분 진폭
    eps = rng.uniform(0, 2 * np.pi, size=w.size)  # 랜덤 위상
    we = encounter_frequency(w, chi_deg, ship.U)

    H_h, H_p = rao_heave_pitch(w, chi_deg, ship)
    H_r = rao_roll(w, chi_deg, ship)

    # (n_comp, n_t) 행렬 없이 메모리 절약형 누적합
    zeta = np.zeros_like(t)
    heave = np.zeros_like(t)
    pitch = np.zeros_like(t)
    roll = np.zeros_like(t)
    for i in range(w.size):
        ph = we[i] * t + eps[i]
        zeta += amp[i] * np.cos(ph)
        heave += amp[i] * np.abs(H_h[i]) * np.cos(ph + np.angle(H_h[i]))
        pitch += amp[i] * np.abs(H_p[i]) * np.cos(ph + np.angle(H_p[i]))
        roll += amp[i] * np.abs(H_r[i]) * np.cos(ph + np.angle(H_r[i]))
    return zeta, heave, np.degrees(pitch), np.degrees(roll)


# ---------------------------------------------------------------------------
# 6. 메인
# ---------------------------------------------------------------------------
def main():
    import os

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)

    # --- 해상상태 5 / P-M 스펙트럼 ---
    Hs_target = 3.25                          # Sea State 5 대표 유의파고 [m]
    U_wind = pm_wind_speed_from_Hs(Hs_target)  # 등가 풍속 U19.5

    w = np.linspace(0.20, 2.50, 800)          # 파주파수 격자 [rad/s]
    S = pm_spectrum(w, U_wind)
    m0_wave = np.trapezoid(S, w)
    Hs_calc = 4.0 * np.sqrt(m0_wave)
    w_peak = w[np.argmax(S)]
    Tp = 2 * np.pi / w_peak

    print("=" * 64)
    print("선박 및 해상 조건")
    print("=" * 64)
    print(f"  L = {Ship.L:.1f} m, 배수량 = {Ship.disp_ton:,.0f} ton, "
          f"V = 5 kts ({Ship.U:.2f} m/s), Fn = {Ship.Fn():.3f}")
    print(f"  가정: B = {Ship.B:.1f} m, Cb = {Ship.Cb:.2f} -> T = {Ship.T:.2f} m, "
          f"GM = {Ship.GM:.1f} m")
    print(f"  Roll 고유주기 = {2*np.pi/Ship.roll_natural_freq():.1f} s")
    print(f"  Sea State 5: Hs(목표) = {Hs_target} m, 등가풍속 U19.5 = {U_wind:.2f} m/s"
          f" ({U_wind/0.514444:.1f} kts)")
    print(f"  P-M 스펙트럼 적분 결과: Hs = {Hs_calc:.2f} m, Tp = {Tp:.1f} s")

    headings = [0.0, 30.0]                    # 입사각: 0°=head sea, 30°
    t = np.arange(0.0, 1800.0, 0.25)          # 30분, dt=0.25 s

    results = {}
    for chi in headings:
        we = encounter_frequency(w, chi, Ship.U)
        H_h, H_p = rao_heave_pitch(w, chi)
        H_r = rao_roll(w, chi)

        stats = {}
        for name, H, scale, unit in [
            ("heave", H_h, 1.0, "m"),
            ("pitch", H_p, 180 / np.pi, "deg"),
            ("roll", H_r, 180 / np.pi, "deg"),
        ]:
            Sr, _ = response_statistics(w, S, H)
            m0, m2 = spectral_moments_encounter(w, we, Sr)
            rms = np.sqrt(m0) * scale
            sig = 2.0 * np.sqrt(m0) * scale   # 유의 단진폭
            Tz = 2 * np.pi * np.sqrt(m0 / m2) if m2 > 0 else np.nan
            stats[name] = dict(Sr=Sr, rms=rms, sig=sig, Tz=Tz, unit=unit)

        zeta, heave, pitch, roll = make_time_series(w, S, chi, t)
        results[chi] = dict(we=we, H_h=H_h, H_p=H_p, H_r=H_r, stats=stats,
                            zeta=zeta, heave=heave, pitch=pitch, roll=roll)

        print("-" * 64)
        print(f"입사각 chi = {chi:.0f} deg "
              f"({'head sea' if chi == 0 else 'bow sea'}) — 주파수영역 통계")
        print("-" * 64)
        print(f"  {'응답':<8}{'RMS':>12}{'유의 단진폭':>16}{'Tz [s]':>10}")
        for name in ("heave", "pitch", "roll"):
            st = stats[name]
            print(f"  {name:<8}{st['rms']:>10.3f} {st['unit']:<3}"
                  f"{st['sig']:>12.3f} {st['unit']:<3}{st['Tz']:>10.2f}")
        # 시계열 검증: 시계열 RMS 와 스펙트럼 RMS 비교
        print(f"  [검증] 시계열 RMS: wave {np.std(zeta):.3f} m "
              f"(스펙트럼 {np.sqrt(m0_wave):.3f} m), "
              f"heave {np.std(heave):.3f} m, pitch {np.std(pitch):.3f} deg, "
              f"roll {np.std(roll):.3f} deg")

    # ------------------------------------------------------------------
    # 그림 1: 주파수영역 (스펙트럼, RAO, 응답스펙트럼)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Frequency-domain results — L=100 m, 10,000 t, 5 kts, Sea State 5 "
        "(P-M spectrum, Jensen et al. 2004 closed-form RAO)",
        fontsize=11,
    )
    colors = {0.0: "tab:blue", 30.0: "tab:red"}

    ax = axes[0, 0]
    ax.plot(w, S, "k-")
    ax.set_title(f"P-M wave spectrum (Hs={Hs_calc:.2f} m, Tp={Tp:.1f} s)")
    ax.set_ylabel(r"$S_\zeta(\omega)$ [m$^2$s]")

    ax = axes[0, 1]
    for chi in headings:
        ax.plot(w, np.abs(results[chi]["H_h"]), color=colors[chi],
                label=f"$\\chi$={chi:.0f}$^\\circ$")
    ax.set_title("Heave RAO")
    ax.set_ylabel("|H| [m/m]")
    ax.legend()

    ax = axes[0, 2]
    for chi in headings:
        ax.plot(w, np.degrees(np.abs(results[chi]["H_p"])), color=colors[chi],
                label=f"$\\chi$={chi:.0f}$^\\circ$")
        ax.plot(w, np.degrees(np.abs(results[chi]["H_r"])), "--",
                color=colors[chi], alpha=0.7,
                label=f"roll, $\\chi$={chi:.0f}$^\\circ$")
    ax.set_title("Pitch RAO (solid) / Roll RAO (dashed)")
    ax.set_ylabel("|H| [deg/m]")
    ax.legend(fontsize=8)

    titles = ["Heave response spectrum", "Pitch response spectrum",
              "Roll response spectrum"]
    keys = ["heave", "pitch", "roll"]
    units = [r"m$^2$s", r"deg$^2$s", r"deg$^2$s"]
    for j, (ttl, key, un) in enumerate(zip(titles, keys, units)):
        ax = axes[1, j]
        for chi in headings:
            st = results[chi]["stats"][key]
            scale = (180 / np.pi) ** 2 if key != "heave" else 1.0
            ax.plot(w, st["Sr"] * scale, color=colors[chi],
                    label=(f"$\\chi$={chi:.0f}$^\\circ$: "
                           f"$2\\sqrt{{m_0}}$={st['sig']:.2f} {st['unit']}"))
        ax.set_title(ttl)
        ax.set_ylabel(f"$S_r(\\omega)$ [{un}]")
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel(r"wave frequency $\omega$ [rad/s]")
        ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f1 = f"{outdir}/frequency_domain.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 그림 2, 3: 시계열 (입사각별)
    # ------------------------------------------------------------------
    tview = (t >= 300) & (t <= 900)           # 300~900 s 구간 표시
    fnames = [f1]
    for chi in headings:
        r = results[chi]
        fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
        lab = "head sea" if chi == 0 else "bow sea"
        fig.suptitle(
            f"Time-domain simulation — incidence $\\chi$={chi:.0f}$^\\circ$ "
            f"({lab}), 5 kts, Sea State 5", fontsize=12)
        for ax, y, name, unit, c in zip(
            axes,
            [r["zeta"], r["heave"], r["pitch"], r["roll"]],
            ["Wave elevation", "Heave", "Pitch", "Roll"],
            ["m", "m", "deg", "deg"],
            ["tab:gray", "tab:blue", "tab:green", "tab:red"],
        ):
            ax.plot(t[tview], y[tview], color=c, lw=0.8)
            ax.set_ylabel(f"{name} [{unit}]")
            ax.grid(alpha=0.3)
            ax.axhline(0, color="k", lw=0.5)
            rms = np.std(y)
            ax.text(0.995, 0.92, f"RMS = {rms:.3f} {unit}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(fc="white", alpha=0.7, ec="none"))
        axes[-1].set_xlabel("time [s]")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fn = f"{outdir}/time_series_chi{int(chi):02d}.png"
        fig.savefig(fn, dpi=140)
        plt.close(fig)
        fnames.append(fn)

    print("=" * 64)
    print("저장된 그림:")
    for fn in fnames:
        print("  ", fn)


if __name__ == "__main__":
    main()
