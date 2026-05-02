"""
주식 분석 시스템 (Stock Analysis System)
- yfinance / FinanceDataReader 기반 데이터 수집
- 기술적 지표 계산 (MA, RSI, MACD)
- Plotly 인터랙티브 대시보드
- 골든크로스 전략 백테스팅
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import math
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BENCHMARKS = ("SPY", "QQQ")
KOREA_BENCHMARKS = ("EWY", "SPY")
CRYPTO_BENCHMARKS = ("BTC-USD", "ETH-USD")

# ── 의존성: import 실패를 None으로 보존 → 실행 시점에만 에러 발생 ──
# (import 시점 sys.exit 제거 → 순수 함수 단위 테스트·부분 import 가능)
try:
    import yfinance as yf
    _yfinance_ok = True
except ImportError:
    yf = None
    _yfinance_ok = False

try:
    import FinanceDataReader as fdr
    _fdr_ok = True
except ImportError:
    fdr = None
    _fdr_ok = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _plotly_ok = True
except ImportError:
    go = None
    make_subplots = None
    _plotly_ok = False


def _check_deps(need_plotly: bool = True, need_fdr: bool = False) -> None:
    """
    실행 시점에 필수 의존성을 확인한다.
    import 시점이 아닌 실제 사용 직전에 호출해 에러를 늦춤으로써
    계산 함수만 import해 테스트하는 경우를 허용한다.
    """
    missing: list[str] = []
    if not _yfinance_ok:
        missing.append("yfinance")
    if need_plotly and not _plotly_ok:
        missing.append("plotly")
    if need_fdr and not _fdr_ok:
        missing.append("finance-datareader")
    if missing:
        raise ImportError(
            f"필수 라이브러리 미설치: {', '.join(missing)}\n"
            f"  → pip install {' '.join(missing)}"
        )


# ──────────────────────────────────────────────
# 공통 헬퍼
# ──────────────────────────────────────────────

def _validate_params(ticker: str,
                     initial_capital: float,
                     commission: float,
                     period_years: float) -> None:
    """핵심 파라미터 사전 검증. 문제가 있으면 ValueError를 발생시킨다."""
    if not ticker or not ticker.strip():
        raise ValueError("ticker는 빈 문자열일 수 없습니다.")
    if initial_capital <= 0:
        raise ValueError(
            f"initial_capital은 0보다 커야 합니다. (입력값: {initial_capital:,})"
        )
    if not (0 <= commission < 1):
        raise ValueError(
            f"commission은 0 이상 1 미만이어야 합니다. (입력값: {commission})"
        )
    if period_years <= 0:
        raise ValueError(
            f"period_years는 0보다 커야 합니다. (입력값: {period_years})"
        )


def _detect_annualization_days(df: pd.DataFrame) -> int:
    """
    데이터에 주말 행이 있으면 365, 없으면 252를 반환한다.

    - 주식(평일 거래): 252
    - 암호화폐·24/7 자산: 365
    """
    sample = df.index[:min(60, len(df))]
    return 365 if any(d.weekday() >= 5 for d in sample) else 252


def _parse_benchmarks(value: str) -> tuple[str, ...]:
    """Parse comma-separated benchmark tickers while preserving order."""
    items = []
    for item in value.split(","):
        benchmark = item.strip().upper()
        if benchmark and benchmark not in items:
            items.append(benchmark)

    if not items:
        raise argparse.ArgumentTypeError("benchmarks must include at least one ticker")

    return tuple(items)


def _default_benchmarks_for_ticker(ticker: str) -> tuple[str, ...]:
    """Choose a small default benchmark set by asset type."""
    normalized = ticker.strip().upper()
    if is_korean_ticker(normalized):
        return KOREA_BENCHMARKS
    if normalized.endswith("-USD"):
        return tuple(item for item in CRYPTO_BENCHMARKS if item != normalized) or DEFAULT_BENCHMARKS
    return DEFAULT_BENCHMARKS


def _make_html_path(ticker: str, overwrite: bool = False) -> Path:
    """
    대시보드 HTML 저장 경로를 결정한다.

    overwrite=True  → <ticker>_dashboard.html  (기존 파일 덮어쓰기)
    overwrite=False → <ticker>_YYYYMMDD_HHMMSS_dashboard.html (타임스탬프)
    """
    safe = ticker.replace("/", "_").replace(":", "_")
    if overwrite:
        fname = f"{safe}_dashboard.html"
    else:
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{safe}_{ts}_dashboard.html"
    return Path(fname)


# ──────────────────────────────────────────────
# 1단계. 데이터 수집
# ──────────────────────────────────────────────

def is_korean_ticker(ticker: str) -> bool:
    """6자리 숫자이면 한국 종목으로 판단."""
    return ticker.isdigit() and len(ticker) == 6


def fetch_data(ticker: str, period_years: float = 1.0,
               start: str = None, end: str = None) -> pd.DataFrame:
    """
    주가 데이터를 수집하여 DataFrame 반환.

    Parameters
    ----------
    ticker       : 종목 코드 (예: AAPL, 005930)
    period_years : 수집 기간(년). start/end 미지정 시 사용.
    start        : 시작일 'YYYY-MM-DD'
    end          : 종료일 'YYYY-MM-DD'

    Returns
    -------
    DataFrame (index=DatetimeIndex, columns=[Open,High,Low,Close,Volume,Adj Close])
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.today() - timedelta(days=int(period_years * 365))).strftime("%Y-%m-%d")

    # 날짜 유효성 검사
    try:
        s_dt = datetime.strptime(start, "%Y-%m-%d")
        e_dt = datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"날짜 형식 오류: start={start}, end={end}. 형식: YYYY-MM-DD")

    if s_dt >= e_dt:
        raise ValueError(f"시작일({start})이 종료일({end})보다 같거나 늦습니다.")

    ticker = ticker.strip()
    # [FIX] 글로벌 종목은 대문자 강제 (yfinance는 소문자 티커 실패)
    if not is_korean_ticker(ticker):
        ticker = ticker.upper()

    # 한국 주식
    if is_korean_ticker(ticker):
        if fdr is None:
            raise ImportError("FinanceDataReader 미설치. pip install finance-datareader")
        df = _fetch_korean(ticker, start, end)
    else:
        df = _fetch_global(ticker, start, end)

    if df is None or df.empty:
        raise ValueError(f"'{ticker}' 에 대한 데이터를 가져올 수 없습니다. 티커를 확인하세요.")

    # 컬럼 정규화
    df = _normalize_columns(df)

    # [FIX] DatetimeIndex 보장
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # 결측값 처리: forward fill 전 현황 경고
    nan_count = df[["Open", "High", "Low", "Close"]].isna().sum().sum()
    if nan_count > 0:
        print(f"[경고] {nan_count}개 결측값 감지 → forward fill 적용")
    df = df.ffill().dropna()

    if len(df) < 30:
        raise ValueError(f"'{ticker}' 데이터가 너무 짧습니다 ({len(df)}일). 기간을 늘려보세요.")

    print(f"[데이터 수집 완료] {ticker} | {df.index[0].date()} ~ {df.index[-1].date()} | {len(df)}거래일")
    return df


def _fetch_korean(ticker: str, start: str, end: str) -> pd.DataFrame:
    """FinanceDataReader로 한국 주식 수집."""
    try:
        df = fdr.DataReader(ticker, start, end)
        return df
    except Exception as e:
        raise ValueError(f"한국 주식 데이터 수집 실패 ({ticker}): {e}")


def _fetch_global(ticker: str, start: str, end: str) -> pd.DataFrame:
    """yfinance로 글로벌 주식 수집."""
    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=False, progress=False)
        if raw.empty:
            raise ValueError(f"yfinance: '{ticker}' 데이터 없음")

        # [FIX] MultiIndex 평탄화 후 중복 컬럼 제거
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.loc[:, ~raw.columns.duplicated(keep="first")]

        return raw
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"글로벌 주식 데이터 수집 실패 ({ticker}): {e}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 표준 형식으로 통일."""
    rename_map = {}
    for col in df.columns:
        # [FIX] strip()으로 앞뒤 공백 제거
        col_lower = str(col).lower().strip().replace(" ", "_")
        if "open" in col_lower:
            rename_map[col] = "Open"
        elif "high" in col_lower:
            rename_map[col] = "High"
        elif "low" in col_lower:
            rename_map[col] = "Low"
        elif "adj" in col_lower and "close" in col_lower:
            # 반드시 "close" 단독 체크보다 먼저 처리
            rename_map[col] = "Adj Close"
        elif "close" in col_lower:
            rename_map[col] = "Close"
        elif "volume" in col_lower or col_lower in ("vol",):
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    for r in required:
        if r not in df.columns:
            if r == "Volume":
                df["Volume"] = 0
            else:
                raise ValueError(f"필수 컬럼 '{r}' 없음. 사용 가능한 컬럼: {list(df.columns)}")

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    return df[["Open", "High", "Low", "Close", "Volume", "Adj Close"]]


# ──────────────────────────────────────────────
# 2단계. 수익률 / 리스크 지표 계산
# ──────────────────────────────────────────────

def compute_returns(df: pd.DataFrame,
                    risk_free_rate: float = 0.03,
                    annualization_days: int = 252) -> dict:
    """
    일간 수익률, 누적 수익률, 변동성, 샤프 비율, MDD 계산.

    Parameters
    ----------
    annualization_days : 연환산 기준 거래일 수.
        주식 = 252, 암호화폐(24/7) = 365.
        _detect_annualization_days(df) 로 자동 결정 권장.

    Returns
    -------
    dict {
        'daily_return': Series,
        'cumulative_return': Series,
        'annual_volatility': float,
        'sharpe_ratio': float,
        'mdd': float,
        'annual_return': float,
        'annualization_days': int,
    }
    """
    close = df["Close"]

    # NaN 유지: std 계산에서 인위적 0 수익률 제외
    daily_ret = close.pct_change()

    # 누적 수익률은 첫 날 NaN → 0으로 채워 cumprod 계산
    cum_ret = (1 + daily_ret.fillna(0)).cumprod()

    # 실제 수익률 개수 (첫 NaN 제외) 로 연환산
    n_days = max(len(daily_ret.dropna()), 1)

    final_cum = cum_ret.iloc[-1]
    if final_cum > 0 and n_days > 0:
        annual_ret = (final_cum ** (annualization_days / n_days)) - 1
    else:
        annual_ret = 0.0

    # NaN 제외한 실제 수익률로 std 계산
    annual_vol = daily_ret.std() * math.sqrt(annualization_days)

    # [FIX] 극소값 방어 (부동소수점 오차 대비)
    if annual_vol > 1e-8:
        sharpe = (annual_ret - risk_free_rate) / annual_vol
    else:
        sharpe = 0.0

    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    mdd = drawdown.min()  # 음수값. 리포트에서 *100하면 음수 퍼센트

    return {
        "daily_return":       daily_ret,
        "cumulative_return":  cum_ret,
        "annual_volatility":  annual_vol,
        "sharpe_ratio":       sharpe,
        "mdd":                mdd,
        "annual_return":      annual_ret,
        "annualization_days": annualization_days,
    }


# ──────────────────────────────────────────────
# 3단계. 기술적 지표
# ──────────────────────────────────────────────

def compute_benchmark_comparison(target_stats: dict,
                                 start: str,
                                 end: str,
                                 risk_free_rate: float = 0.03,
                                 annualization_days: int = 252,
                                 benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS) -> list[dict]:
    """
    Compare target return metrics with simple market benchmarks.

    Benchmark fetch failures are kept in the returned rows so the main analysis
    can continue even when an external data source is temporarily unavailable.
    """
    target_total_return = float(target_stats["cumulative_return"].iloc[-1] - 1)
    rows: list[dict] = []

    for benchmark in benchmarks:
        try:
            bm_df = fetch_data(benchmark, start=start, end=end)
            bm_stats = compute_returns(
                bm_df,
                risk_free_rate=risk_free_rate,
                annualization_days=annualization_days,
            )
            total_return = float(bm_stats["cumulative_return"].iloc[-1] - 1)
            rows.append({
                "ticker": benchmark,
                "total_return": total_return,
                "annual_return": float(bm_stats["annual_return"]),
                "mdd": float(bm_stats["mdd"]),
                "excess_return": target_total_return - total_return,
                "cumulative_return": bm_stats["cumulative_return"],
                "error": None,
            })
        except Exception as exc:
            rows.append({
                "ticker": benchmark,
                "total_return": None,
                "annual_return": None,
                "mdd": None,
                "excess_return": None,
                "cumulative_return": None,
                "error": str(exc),
            })

    return rows


def print_benchmark_report(ticker: str, target_stats: dict, benchmark_rows: list[dict]) -> None:
    """Print a compact benchmark comparison table."""
    target_total_return = float(target_stats["cumulative_return"].iloc[-1] - 1)
    sep = "-" * 60

    print(f"\n[ 벤치마크 비교 | 기준: {ticker} ]")
    print(sep)
    print(f"  {'대상':<8} {'총수익률':>10} {'연환산':>10} {'MDD':>10} {'초과수익':>10}")
    print(sep)
    print(f"  {ticker:<8} {target_total_return*100:>9.2f}% "
          f"{target_stats['annual_return']*100:>9.2f}% "
          f"{target_stats['mdd']*100:>9.2f}% {'-':>10}")

    for row in benchmark_rows:
        if row["error"]:
            print(f"  {row['ticker']:<8} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            print(f"    - 비교 불가: {row['error']}")
            continue

        print(f"  {row['ticker']:<8} {row['total_return']*100:>9.2f}% "
              f"{row['annual_return']*100:>9.2f}% "
              f"{row['mdd']*100:>9.2f}% "
              f"{row['excess_return']*100:>+9.2f}%")


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """5/20/60/120일 이동평균선 및 골든/데드 크로스 시그널 추가."""
    close = df["Close"]
    df["MA5"]   = close.rolling(window=5,   min_periods=5).mean()
    df["MA20"]  = close.rolling(window=20,  min_periods=20).mean()
    df["MA60"]  = close.rolling(window=60,  min_periods=60).mean()
    df["MA120"] = close.rolling(window=120, min_periods=120).mean()

    # NaN 구간에서는 비교가 False → 시그널 없음 (의도된 동작)
    cond_golden = (df["MA20"] > df["MA60"]) & (df["MA20"].shift(1) <= df["MA60"].shift(1))
    cond_dead   = (df["MA20"] < df["MA60"]) & (df["MA20"].shift(1) >= df["MA60"].shift(1))
    df["MA_Signal"] = np.where(cond_golden, "golden", np.where(cond_dead, "dead", ""))
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI 계산 (Wilder's smoothing) 및 buy/sell/neutral 시그널 추가.

    Wilder's RSI: alpha = 1/period (ewm com = period-1)
    """
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # [FIX] 세 가지 케이스 명확 분리
    #   avg_gain > 0, avg_loss = 0 → RSI = 100 (상승만 존재)
    #   avg_gain = 0, avg_loss = 0 → RSI = 50  (flat price, 중립)
    #   avg_gain = 0, avg_loss > 0 → RSI = 0   (하락만 존재, 정상 공식)
    #   NaN (워밍업 미완료)       → RSI = NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        both_zero = (avg_gain == 0) & (avg_loss == 0)
        rs = np.where(avg_loss == 0, np.inf, avg_gain.values / avg_loss.values)
    rsi_vals = np.where(
        both_zero,
        50.0,                                          # flat price → 중립
        np.where(np.isinf(rs), 100.0,
                 100.0 - (100.0 / (1.0 + rs)))
    )
    rsi = pd.Series(rsi_vals, index=df.index)
    # 워밍업 구간(avg_gain/avg_loss가 NaN) → RSI도 NaN 유지
    rsi = rsi.where(avg_gain.notna() & avg_loss.notna(), other=np.nan)

    df["RSI"] = rsi
    # [FIX] RSI NaN 구간은 "" 처리 (기존 "neutral" 오표기 방지)
    df["RSI_Signal"] = np.where(
        rsi.isna(), "",
        np.where(rsi >= 70, "sell",
                 np.where(rsi <= 30, "buy", "neutral"))
    )
    return df


def compute_macd(df: pd.DataFrame,
                 fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD Line, Signal Line, Histogram 및 크로스 시그널 추가."""
    ema_fast    = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow    = df["Close"].ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line

    df["MACD"]        = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"]   = histogram

    cond_golden = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    cond_dead   = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    df["MACD_Cross"] = np.where(cond_golden, "golden", np.where(cond_dead, "dead", ""))
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """모든 기술적 지표를 DataFrame에 추가."""
    df = df.copy()
    df = compute_moving_averages(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    return df


def summarize_current_state(df: pd.DataFrame) -> dict:
    """
    최신 행 기준으로 현재 기술적 상태를 요약한다.

    투자 판단을 대신하지 않고, 차트 해석에 필요한 추세/모멘텀 상태를
    빠르게 확인하기 위한 보조 정보다.
    """
    latest = df.iloc[-1]
    close = float(latest["Close"])
    ma20  = latest.get("MA20", np.nan)
    ma60  = latest.get("MA60", np.nan)
    ma120 = latest.get("MA120", np.nan)
    rsi   = latest.get("RSI", np.nan)
    macd  = latest.get("MACD", np.nan)
    signal = latest.get("MACD_Signal", np.nan)

    price_position = {
        "above_ma20":  bool(pd.notna(ma20) and close > ma20),
        "above_ma60":  bool(pd.notna(ma60) and close > ma60),
        "above_ma120": bool(pd.notna(ma120) and close > ma120),
    }

    if price_position["above_ma20"] and price_position["above_ma60"]:
        trend = "상승 우위"
    elif not price_position["above_ma20"] and not price_position["above_ma60"]:
        trend = "하락 우위"
    else:
        trend = "중립/전환 구간"

    if pd.isna(rsi):
        rsi_state = "판단 보류"
    elif rsi >= 70:
        rsi_state = "과매수"
    elif rsi <= 30:
        rsi_state = "과매도"
    elif rsi >= 50:
        rsi_state = "중립~강세"
    else:
        rsi_state = "중립~약세"

    if pd.isna(macd) or pd.isna(signal):
        macd_state = "판단 보류"
    elif macd > signal:
        macd_state = "상승 모멘텀"
    else:
        macd_state = "하락/둔화 모멘텀"

    score_checks = [
        price_position["above_ma20"],
        price_position["above_ma60"],
        bool(pd.notna(ma20) and pd.notna(ma60) and ma20 > ma60),
        bool(pd.notna(macd) and pd.notna(signal) and macd > signal),
        bool(pd.notna(rsi) and 40 <= rsi <= 70),
    ]
    signal_score = sum(score_checks)

    return {
        "date": df.index[-1],
        "close": close,
        "trend": trend,
        "price_position": price_position,
        "rsi": None if pd.isna(rsi) else float(rsi),
        "rsi_state": rsi_state,
        "macd": None if pd.isna(macd) else float(macd),
        "macd_signal": None if pd.isna(signal) else float(signal),
        "macd_state": macd_state,
        "signal_score": signal_score,
        "signal_score_max": len(score_checks),
    }


def print_current_state(summary: dict, ticker: str) -> None:
    """현재 기술적 상태 요약을 콘솔에 출력."""
    date = summary["date"].strftime("%Y-%m-%d") if hasattr(summary["date"], "strftime") else str(summary["date"])
    price = summary["price_position"]
    rsi_text = "N/A" if summary["rsi"] is None else f"{summary['rsi']:.1f}"

    print(f"\n[ 현재 상태 요약 | {ticker} | {date} ]")
    print(f"  종가              : {summary['close']:,.2f}")
    print(f"  추세              : {summary['trend']}")
    print("  가격 위치         : "
          f"MA20 {'위' if price['above_ma20'] else '아래'}, "
          f"MA60 {'위' if price['above_ma60'] else '아래'}, "
          f"MA120 {'위' if price['above_ma120'] else '아래'}")
    print(f"  RSI               : {rsi_text} ({summary['rsi_state']})")
    print(f"  MACD              : {summary['macd_state']}")
    print(f"  종합 점수         : {summary['signal_score']} / {summary['signal_score_max']}")


# ──────────────────────────────────────────────
# 4단계. 시각화
# ──────────────────────────────────────────────

def _format_current_state_annotation(summary: dict) -> str:
    """
    Return a compact HTML-safe summary for the dashboard header area.
    """
    rsi = summary.get("rsi")
    rsi_text = "N/A" if rsi is None or pd.isna(rsi) else f"{rsi:.1f}"
    return (
        f"<b>현재 상태</b>: {summary.get('trend', 'N/A')} | "
        f"RSI {rsi_text} ({summary.get('rsi_state', 'N/A')}) | "
        f"MACD {summary.get('macd_state', 'N/A')} | "
        f"점수 {summary.get('signal_score', 0)}/{summary.get('signal_score_max', 0)}"
    )


def _format_benchmark_annotation(benchmark_rows: list[dict] | None) -> str | None:
    """Return a compact benchmark comparison summary for the dashboard."""
    if not benchmark_rows:
        return None

    parts = []
    for row in benchmark_rows:
        if row.get("error"):
            parts.append(f"{row['ticker']} N/A")
        else:
            parts.append(f"{row['ticker']} 초과 {row['excess_return']*100:+.2f}%")

    return "<b>벤치마크</b>: " + " | ".join(parts)


def plot_dashboard(df: pd.DataFrame,
                   ticker: str,
                   stats: dict,
                   current_state: dict | None = None,
                   benchmark_comparison: list[dict] | None = None):
    """
    Plotly 서브플롯 대시보드 (캔들 + MA, 거래량, RSI, MACD).
    """
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        row_heights=[0.38, 0.13, 0.17, 0.17, 0.15],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{ticker} 캔들스틱 + 이동평균선",
            "거래량",
            "RSI (14)",
            "MACD (12/26/9)",
            "누적수익률 비교"
        ]
    )

    # ── 차트1: 캔들스틱 ──
    # [FIX] fillcolor 명시 (line color만 설정 시 캔들 몸통이 기본값으로 표시되는 문제)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        name="캔들",
        increasing_line_color="#EF5350",
        increasing_fillcolor="rgba(239,83,80,0.85)",
        decreasing_line_color="#26A69A",
        decreasing_fillcolor="rgba(38,166,154,0.85)",
    ), row=1, col=1)

    # [FIX] MA120 색상을 #EF5350(캔들 상승색)과 구분되게 변경
    ma_colors = {
        "MA5":   "#FFA726",  # 주황
        "MA20":  "#42A5F5",  # 파랑
        "MA60":  "#AB47BC",  # 보라
        "MA120": "#EC407A",  # 핑크 (기존 빨강 → 캔들 상승색과 혼동 방지)
    }
    for ma, color in ma_colors.items():
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ma], name=ma,
                line=dict(color=color, width=1.2), opacity=0.85
            ), row=1, col=1)

    # [FIX] 골든/데드 크로스 마커: NaN MA20 행 제거 후 표시
    golden = df[(df["MA_Signal"] == "golden") & df["MA20"].notna()]
    dead   = df[(df["MA_Signal"] == "dead")   & df["MA20"].notna()]
    if not golden.empty:
        fig.add_trace(go.Scatter(
            x=golden.index, y=golden["MA20"],
            mode="markers", marker=dict(symbol="triangle-up", size=12, color="#00E676"),
            name="골든크로스"
        ), row=1, col=1)
    if not dead.empty:
        fig.add_trace(go.Scatter(
            x=dead.index, y=dead["MA20"],
            mode="markers", marker=dict(symbol="triangle-down", size=12, color="#FF1744"),
            name="데드크로스"
        ), row=1, col=1)

    # ── 차트2: 거래량 ──
    vol_colors = ["#EF5350" if c >= o else "#26A69A"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="거래량",
        marker_color=vol_colors, opacity=0.7
    ), row=2, col=1)

    # ── 차트3: RSI ──
    # [FIX] NaN 구간 제거하여 그래프 끊김 방지
    rsi_valid = df["RSI"].dropna()
    fig.add_trace(go.Scatter(
        x=rsi_valid.index, y=rsi_valid, name="RSI",
        line=dict(color="#FF9800", width=1.5)
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",  opacity=0.6, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="blue", opacity=0.6, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot",  line_color="gray", opacity=0.4, row=3, col=1)

    # [FIX] RSI 시그널 마커: RSI NaN 제외
    rsi_buy  = df[(df["RSI_Signal"] == "buy")  & df["RSI"].notna()]
    rsi_sell = df[(df["RSI_Signal"] == "sell") & df["RSI"].notna()]
    if not rsi_buy.empty:
        fig.add_trace(go.Scatter(
            x=rsi_buy.index, y=rsi_buy["RSI"],
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="#00BCD4"),
            name="RSI 매수"
        ), row=3, col=1)
    if not rsi_sell.empty:
        fig.add_trace(go.Scatter(
            x=rsi_sell.index, y=rsi_sell["RSI"],
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="#E91E63"),
            name="RSI 매도"
        ), row=3, col=1)

    # ── 차트4: MACD ──
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], name="MACD",
        line=dict(color="#2196F3", width=1.5)
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"], name="Signal",
        line=dict(color="#FF5722", width=1.5)
    ), row=4, col=1)

    hist_colors = ["#EF5350" if v >= 0 else "#26A69A"
                   for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"], name="히스토그램",
        marker_color=hist_colors, opacity=0.6
    ), row=4, col=1)

    # [FIX] MACD 크로스 마커: NaN MACD 행 제거
    macd_golden = df[(df["MACD_Cross"] == "golden") & df["MACD"].notna()]
    macd_dead   = df[(df["MACD_Cross"] == "dead")   & df["MACD"].notna()]
    if not macd_golden.empty:
        fig.add_trace(go.Scatter(
            x=macd_golden.index, y=macd_golden["MACD"],
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="#00E676"),
            name="MACD 골든"
        ), row=4, col=1)
    if not macd_dead.empty:
        fig.add_trace(go.Scatter(
            x=macd_dead.index, y=macd_dead["MACD"],
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="#FF1744"),
            name="MACD 데드"
        ), row=4, col=1)

    # ── 레이아웃 ──
    target_cum_pct = (stats["cumulative_return"] - 1) * 100
    fig.add_trace(go.Scatter(
        x=target_cum_pct.index,
        y=target_cum_pct,
        name=f"{ticker} 누적수익률",
        line=dict(color="#FFFFFF", width=2.0),
    ), row=5, col=1)

    benchmark_colors = ["#00BCD4", "#FFD54F", "#A5D6A7", "#CE93D8"]
    for idx, row in enumerate(benchmark_comparison or []):
        series = row.get("cumulative_return")
        if row.get("error") or series is None:
            continue
        fig.add_trace(go.Scatter(
            x=series.index,
            y=(series - 1) * 100,
            name=f"{row['ticker']} 누적수익률",
            line=dict(color=benchmark_colors[idx % len(benchmark_colors)], width=1.5, dash="dot"),
        ), row=5, col=1)

    annual_ret_pct = stats["annual_return"]    * 100
    annual_vol_pct = stats["annual_volatility"] * 100
    mdd_pct        = stats["mdd"]              * 100  # 음수

    benchmark_annotation = _format_benchmark_annotation(benchmark_comparison)
    top_margin = 145 if current_state and benchmark_annotation else 120 if current_state or benchmark_annotation else 90

    fig.update_layout(
        title=dict(
            text=(f"<b>{ticker} 주식 분석 대시보드</b>  |  "
                  f"연수익률: {annual_ret_pct:+.2f}%  |  "
                  f"연변동성: {annual_vol_pct:.2f}%  |  "
                  f"Sharpe: {stats['sharpe_ratio']:.2f}  |  "
                  f"MDD: {mdd_pct:.2f}%"),
            font=dict(size=14)
        ),
        height=1120,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=top_margin, b=40)
    )

    fig.update_yaxes(title_text="주가",  row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    fig.update_yaxes(title_text="RSI",   row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",  row=4, col=1)
    fig.update_yaxes(title_text="누적수익률(%)", row=5, col=1)

    # [FIX] 주식만 주말 갭 제거 (암호화폐·24시간 자산은 주말 데이터 존재 → 적용 제외)
    if benchmark_annotation:
        fig.add_annotation(
            text=benchmark_annotation,
            xref="paper",
            yref="paper",
            x=0,
            y=1.04,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            align="left",
            font=dict(size=12, color="#F5F5F5"),
            bgcolor="rgba(20, 24, 31, 0.72)",
            bordercolor="rgba(255, 255, 255, 0.18)",
            borderwidth=1,
            borderpad=6,
        )

    if current_state:
        fig.add_annotation(
            text=_format_current_state_annotation(current_state),
            xref="paper",
            yref="paper",
            x=0,
            y=1.09,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            align="left",
            font=dict(size=12, color="#F5F5F5"),
            bgcolor="rgba(20, 24, 31, 0.78)",
            bordercolor="rgba(255, 255, 255, 0.22)",
            borderwidth=1,
            borderpad=6,
        )

    has_weekend = any(d.weekday() >= 5 for d in df.index[:min(60, len(df))])
    if not has_weekend:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    return fig


# ──────────────────────────────────────────────
# 5단계. 백테스팅
# ──────────────────────────────────────────────

def backtest_golden_cross(df: pd.DataFrame,
                          initial_capital: float = 10_000_000,
                          commission: float = 0.00015,
                          risk_free_rate: float = 0.03,
                          annualization_days: int = 252) -> dict:
    """
    골든크로스(MA20 > MA60) 매수 / 데드크로스 매도 전략 백테스팅.

    Parameters
    ----------
    df                 : 기술적 지표가 추가된 DataFrame
    initial_capital    : 초기 자본금 (원)
    commission         : 수수료율 (기본 0.015%)
    risk_free_rate     : 무위험 수익률 (연, 소수점)
    annualization_days : 연환산 기준 거래일 수 (주식=252, 암호화폐=365)

    Returns
    -------
    dict with trade log and performance metrics
    """
    # [FIX] iterrows → NumPy 배열 기반 반복 (약 10x 빠름)
    signals = df["MA_Signal"].values
    prices  = df["Close"].values
    dates   = df.index

    cash   = initial_capital
    shares = 0.0
    trades = []
    equity = np.empty(len(df))

    for i in range(len(df)):
        # [FIX] row.get() deprecated → 배열 직접 참조
        signal = signals[i]
        price  = float(prices[i])

        if signal == "golden" and cash > 0:
            fee    = cash * commission
            shares = (cash - fee) / price
            trades.append({
                "date": dates[i], "type": "buy",
                "price": price, "shares": shares, "fee": fee
            })
            cash = 0.0

        elif signal == "dead" and shares > 0:
            sell_amount = shares * price
            fee         = sell_amount * commission
            cash        = sell_amount - fee
            trades.append({
                "date": dates[i], "type": "sell",
                "price": price, "shares": shares, "fee": fee
            })
            shares = 0.0

        equity[i] = cash + shares * price

    # 기간 말 잔여 포지션 청산
    if shares > 0:
        last_price  = float(prices[-1])
        sell_amount = shares * last_price
        fee         = sell_amount * commission
        cash        = sell_amount - fee
        trades.append({
            "date": dates[-1], "type": "sell(final)",
            "price": last_price, "shares": shares, "fee": fee
        })
        equity[-1] = cash

    equity_series = pd.Series(equity, index=dates)

    # 성과 지표
    final_equity = float(equity_series.iloc[-1])
    total_ret    = (final_equity - initial_capital) / initial_capital

    # n_days = 실제 수익률 기간 수 (len - 1)
    n_days    = max(len(equity_series) - 1, 1)
    final_cum = final_equity / initial_capital if initial_capital > 0 else 1.0
    annual_ret = (final_cum ** (annualization_days / n_days)) - 1 if final_cum > 0 else 0.0

    daily_ret_bt = equity_series.pct_change().dropna()
    annual_vol   = daily_ret_bt.std() * math.sqrt(annualization_days)

    # [FIX] risk_free_rate 파라미터 사용 (기존 하드코딩 0.03 제거)
    if annual_vol > 1e-8:
        sharpe = (annual_ret - risk_free_rate) / annual_vol
    else:
        sharpe = 0.0

    roll_max  = equity_series.cummax()
    mdd       = ((equity_series - roll_max) / roll_max).min()

    # 승률: buy-sell 페어 매칭 (전략상 항상 교대 발생)
    buy_trades  = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if "sell" in t["type"]]

    wins = sum(
        1 for b, s in zip(buy_trades, sell_trades)
        if s["price"] > b["price"]
    )
    n_trades = len(sell_trades)
    win_rate = wins / n_trades if n_trades > 0 else 0.0

    if n_trades == 0:
        print("[경고] 백테스팅 기간 중 거래 신호 없음. 기간을 늘리거나 종목을 변경하세요.")

    return {
        "trades":          trades,
        "equity_series":   equity_series,
        "final_equity":    final_equity,
        "total_return":    total_ret,
        "annual_return":   annual_ret,
        "annual_vol":      annual_vol,
        "sharpe":          sharpe,
        "mdd":             mdd,
        "n_trades":        n_trades,
        "win_rate":        win_rate,
        "initial_capital": initial_capital,
    }


def backtest_buy_and_hold(df: pd.DataFrame,
                          initial_capital: float = 10_000_000,
                          commission: float = 0.00015,
                          risk_free_rate: float = 0.03,
                          annualization_days: int = 252) -> dict:
    """
    Buy & Hold 전략 성과 계산.

    Parameters
    ----------
    risk_free_rate     : 무위험 수익률 (연, 소수점)
    annualization_days : 연환산 기준 거래일 수 (주식=252, 암호화폐=365)
    """
    buy_price = float(df["Close"].iloc[0])
    buy_fee   = initial_capital * commission
    shares    = (initial_capital - buy_fee) / buy_price

    final_price  = float(df["Close"].iloc[-1])
    sell_amount  = shares * final_price
    sell_fee     = sell_amount * commission
    final_equity = sell_amount - sell_fee

    total_ret = (final_equity - initial_capital) / initial_capital

    # n_days = 실제 수익률 기간 수 (len - 1)
    n_days    = max(len(df) - 1, 1)
    annual_ret = (1 + total_ret) ** (annualization_days / n_days) - 1 \
                 if (1 + total_ret) > 0 else 0.0

    close_ret  = df["Close"].pct_change().dropna()
    annual_vol = close_ret.std() * math.sqrt(annualization_days)

    # [FIX] risk_free_rate 파라미터 사용
    if annual_vol > 1e-8:
        sharpe = (annual_ret - risk_free_rate) / annual_vol
    else:
        sharpe = 0.0

    cum      = (1 + close_ret).cumprod()
    roll_max = cum.cummax()
    mdd      = ((cum - roll_max) / roll_max).min()

    return {
        "final_equity":    final_equity,
        "total_return":    total_ret,
        "annual_return":   annual_ret,
        "annual_vol":      annual_vol,
        "sharpe":          sharpe,
        "mdd":             mdd,
        "initial_capital": initial_capital,
    }


def print_backtest_report(bt_result: dict, bah_result: dict, ticker: str):
    """백테스팅 결과를 콘솔에 출력."""
    sep = "-" * 60

    print(f"\n{'='*60}")
    print(f"  백테스팅 결과 보고서 | {ticker}")
    print(f"{'='*60}")

    print(f"\n[ 골든크로스 전략 ]")
    print(sep)
    print(f"  초기 자본금       : {bt_result['initial_capital']:>15,.0f} 원")
    print(f"  최종 자산         : {bt_result['final_equity']:>15,.0f} 원")
    print(f"  총 수익률         : {bt_result['total_return']*100:>14.2f} %")
    print(f"  연환산 수익률     : {bt_result['annual_return']*100:>14.2f} %")
    print(f"  연간 변동성       : {bt_result['annual_vol']*100:>14.2f} %")
    print(f"  샤프 비율         : {bt_result['sharpe']:>15.3f}")
    print(f"  최대 낙폭 (MDD)   : {bt_result['mdd']*100:>14.2f} %")
    print(f"  총 거래 횟수      : {bt_result['n_trades']:>15} 회")
    print(f"  승률              : {bt_result['win_rate']*100:>14.1f} %")

    print(f"\n[ Buy & Hold 전략 ]")
    print(sep)
    print(f"  초기 자본금       : {bah_result['initial_capital']:>15,.0f} 원")
    print(f"  최종 자산         : {bah_result['final_equity']:>15,.0f} 원")
    print(f"  총 수익률         : {bah_result['total_return']*100:>14.2f} %")
    print(f"  연환산 수익률     : {bah_result['annual_return']*100:>14.2f} %")
    print(f"  연간 변동성       : {bah_result['annual_vol']*100:>14.2f} %")
    print(f"  샤프 비율         : {bah_result['sharpe']:>15.3f}")
    print(f"  최대 낙폭 (MDD)   : {bah_result['mdd']*100:>14.2f} %")

    # 비교 요약
    print(f"\n[ 전략 비교 요약 ]")
    print(sep)
    diff_ret    = bt_result['total_return'] - bah_result['total_return']
    diff_sharpe = bt_result['sharpe']       - bah_result['sharpe']
    diff_mdd    = bt_result['mdd']          - bah_result['mdd']
    print(f"  수익률 차이       : {diff_ret*100:>+14.2f} %")
    print(f"  샤프 비율 차이    : {diff_sharpe:>+15.3f}")
    print(f"  MDD 차이          : {diff_mdd*100:>+14.2f} %")
    winner = ("골든크로스 전략" if bt_result['total_return'] > bah_result['total_return']
              else "Buy & Hold 전략")
    print(f"\n  > 이 기간 우위 전략: {winner}")

    # [FIX] 개별 거래 내역 (매수/매도 페어 + 손익)
    trades      = bt_result.get("trades", [])
    buy_trades  = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if "sell" in t["type"]]

    if buy_trades:
        print(f"\n[ 거래 내역 상세 ]")
        print(f"  {'#':>3}  {'매수일':^12}  {'매수가':>10}  {'매도일':^12}  {'매도가':>10}  {'손익(%)':>10}  {'결과':^4}")
        print(sep)
        for idx, (b, s) in enumerate(zip(buy_trades, sell_trades), start=1):
            buy_d  = b["date"].strftime("%Y-%m-%d") if hasattr(b["date"], "strftime") else str(b["date"])[:10]
            sell_d = s["date"].strftime("%Y-%m-%d") if hasattr(s["date"], "strftime") else str(s["date"])[:10]
            net_pnl_pct = (s["price"] - b["price"]) / b["price"] * 100
            result      = "WIN " if s["price"] > b["price"] else "LOSS"
            print(f"  {idx:>3}  {buy_d:^12}  {b['price']:>10,.1f}  {sell_d:^12}  {s['price']:>10,.1f}  {net_pnl_pct:>+9.2f}%  {result:^4}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# 분석 파이프라인 통합
# ──────────────────────────────────────────────

def run_analysis(ticker: str,
                 period_years: float = 1.0,
                 start: str = None,
                 end: str = None,
                 initial_capital: float = 10_000_000,
                 commission: float = 0.00015,
                 risk_free_rate: float = 0.03,
                 show_chart: bool = True,
                 overwrite_html: bool = False,
                 benchmarks: tuple[str, ...] | None = None) -> dict:
    """
    전체 분석 파이프라인 실행.

    Parameters
    ----------
    ticker          : 종목 코드 (예: AAPL, 005930, BTC-USD)
    period_years    : 분석 기간 (년, start/end 미지정 시 사용)
    start / end     : 직접 날짜 지정 'YYYY-MM-DD' (None이면 period_years 사용)
    initial_capital : 백테스팅 초기 자본금
    commission      : 매수·매도 수수료율 (기본 0.015%)
    risk_free_rate  : 무위험 수익률 (연, 소수점)
    show_chart      : Plotly 대시보드 표시 여부
    overwrite_html  : True → 기존 HTML 덮어쓰기 / False → 타임스탬프 파일명

    Returns
    -------
    dict { df, stats, bt_result, bah_result, annualization_days }
    """
    # 0. 파라미터 검증 & 의존성 확인
    _validate_params(ticker, initial_capital, commission, period_years)
    _check_deps(need_plotly=show_chart,
                need_fdr=ticker.isdigit() and len(ticker) == 6)

    print(f"\n{'='*55}")
    print(f"  주식 분석 시스템 시작 | 티커: {ticker}")
    print(f"{'='*55}")

    # 1. 데이터 수집
    df = fetch_data(ticker, period_years=period_years, start=start, end=end)

    # 2. 자산 유형별 연환산 기준 자동 결정
    ann_days = _detect_annualization_days(df)
    print(f"[연환산 기준] {ann_days}일 "
          f"({'암호화폐·24/7' if ann_days == 365 else '주식·평일 거래'})")

    # 3. 수익률/리스크 지표
    stats = compute_returns(df,
                            risk_free_rate=risk_free_rate,
                            annualization_days=ann_days)

    # 4. 기술적 지표
    df = add_indicators(df)
    print(f"[지표 계산 완료] MA(5/20/60/120), RSI(14), MACD(12/26/9)")

    # 5. 현재 상태 요약
    current_state = summarize_current_state(df)
    print_current_state(current_state, ticker)

    selected_benchmarks = benchmarks or _default_benchmarks_for_ticker(ticker)
    benchmark_start = df.index[0].strftime("%Y-%m-%d")
    benchmark_end = (df.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
    benchmark_comparison = compute_benchmark_comparison(
        stats,
        start=benchmark_start,
        end=benchmark_end,
        risk_free_rate=risk_free_rate,
        annualization_days=ann_days,
        benchmarks=selected_benchmarks,
    )
    print_benchmark_report(ticker, stats, benchmark_comparison)

    # 6. 시각화
    if show_chart:
        fig      = plot_dashboard(df, ticker, stats, current_state=current_state,
                                  benchmark_comparison=benchmark_comparison)
        out_path = _make_html_path(ticker, overwrite=overwrite_html)
        fig.write_html(out_path)
        print(f"[차트 저장] {out_path}")
        fig.show()

    # 7. 백테스팅
    bt_result  = backtest_golden_cross(df,
                                       initial_capital=initial_capital,
                                       commission=commission,
                                       risk_free_rate=risk_free_rate,
                                       annualization_days=ann_days)
    bah_result = backtest_buy_and_hold(df,
                                       initial_capital=initial_capital,
                                       commission=commission,
                                       risk_free_rate=risk_free_rate,
                                       annualization_days=ann_days)
    print_backtest_report(bt_result, bah_result, ticker)

    return {"df": df, "stats": stats,
            "bt_result": bt_result, "bah_result": bah_result,
            "annualization_days": ann_days,
            "current_state": current_state,
            "benchmark_comparison": benchmark_comparison}


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────

def main() -> None:
    """
    CLI 진입점.

    파일 상단 설정값 변경 방식과 CLI 인자 방식 모두 지원.
    CLI 인자가 있으면 CLI 우선, 없으면 아래 기본값 사용.

    사용 예시
    ---------
    python stock_analysis.py                          # 기본값(AAPL, 2년)
    python stock_analysis.py TSLA --years 3
    python stock_analysis.py 005930 --capital 5000000
    python stock_analysis.py BTC-USD --years 1 --no-chart
    python stock_analysis.py AAPL --start 2022-01-01 --end 2024-01-01
    python stock_analysis.py AAPL --overwrite        # HTML 덮어쓰기
    """
    # ── 코드 내 기본값 (CLI 인자 없을 때 사용) ──────────────
    DEFAULT_TICKER          = "AAPL"
    DEFAULT_PERIOD_YEARS    = 2.0
    DEFAULT_INITIAL_CAPITAL = 10_000_000
    DEFAULT_COMMISSION      = 0.00015
    DEFAULT_RISK_FREE_RATE  = 0.03
    # ────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        prog="stock_analysis",
        description="주식·암호화폐 기술적 분석 및 골든크로스 백테스팅",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "ticker", nargs="?", default=DEFAULT_TICKER,
        help=f"종목 코드 (기본값: {DEFAULT_TICKER})\n예: AAPL  005930  BTC-USD  TSLA"
    )
    parser.add_argument(
        "--years", type=float, default=DEFAULT_PERIOD_YEARS,
        metavar="N",
        help=f"분석 기간 (년, 기본값: {DEFAULT_PERIOD_YEARS})"
    )
    parser.add_argument(
        "--start", type=str, default=None, metavar="YYYY-MM-DD",
        help="시작일 (지정 시 --years 무시)"
    )
    parser.add_argument(
        "--end", type=str, default=None, metavar="YYYY-MM-DD",
        help="종료일 (지정 시 --years 무시)"
    )
    parser.add_argument(
        "--capital", type=float, default=DEFAULT_INITIAL_CAPITAL,
        metavar="N",
        help=f"초기 자본금 (기본값: {DEFAULT_INITIAL_CAPITAL:,})"
    )
    parser.add_argument(
        "--commission", type=float, default=DEFAULT_COMMISSION,
        metavar="F",
        help=f"수수료율 0~1 (기본값: {DEFAULT_COMMISSION})"
    )
    parser.add_argument(
        "--rfr", type=float, default=DEFAULT_RISK_FREE_RATE,
        metavar="F",
        help=f"무위험 수익률 (기본값: {DEFAULT_RISK_FREE_RATE})"
    )
    parser.add_argument(
        "--no-chart", action="store_true",
        help="차트 표시 및 HTML 저장 생략"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="기존 HTML 대시보드 덮어쓰기 (기본: 타임스탬프 파일명)"
    )

    parser.add_argument(
        "--benchmarks", type=_parse_benchmarks, default=None,
        metavar="A,B",
        help="benchmark tickers separated by commas (default: auto by asset type)"
    )

    args = parser.parse_args()

    try:
        run_analysis(
            ticker=args.ticker,
            period_years=args.years,
            start=args.start,
            end=args.end,
            initial_capital=args.capital,
            commission=args.commission,
            risk_free_rate=args.rfr,
            show_chart=not args.no_chart,
            overwrite_html=args.overwrite,
            benchmarks=args.benchmarks,
        )
    except (ValueError, ImportError) as e:
        print(f"\n[오류] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[예기치 못한 오류] {e}")
        raise


if __name__ == "__main__":
    main()
