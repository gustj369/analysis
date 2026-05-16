"""
주식 분석 시스템 (Stock Analysis System)
- yfinance / FinanceDataReader 기반 데이터 수집
- 기술적 지표 계산 (MA, RSI, MACD)
- Plotly 인터랙티브 대시보드
- 골든크로스 전략 백테스팅
"""

# ── 표준 라이브러리 ───────────────────────────────────────────────────────────
import re
import sys
import math
import argparse
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict

# ── 서드파티 (항상 사용) ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── 경고 필터: 전역 억제 대신 특정 라이브러리·카테고리만 차단 ────────────────
# [FIX 1차 H-1] warnings.filterwarnings("ignore") 전역 설정 → 범위 축소
warnings.filterwarnings("ignore", category=FutureWarning,      module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning,      module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")
warnings.filterwarnings("ignore", category=UserWarning,        module="yfinance")


# ──────────────────────────────────────────────────────────────────────────────
# 모듈 레벨 기본값 상수
# [FIX 1차 H-3] main() 내부 정의 → 모듈 레벨 이동 (CLI·시그니처·로직 공유)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_TICKER          = "AAPL"
DEFAULT_PERIOD_YEARS    = 2.0
DEFAULT_INITIAL_CAPITAL = 10_000_000
DEFAULT_COMMISSION      = 0.00015
DEFAULT_RISK_FREE_RATE  = 0.03

# ── 벤치마크 프리셋 ──────────────────────────────────────────────────────────
DEFAULT_BENCHMARKS = ("SPY", "QQQ")
KOREA_BENCHMARKS   = ("EWY", "SPY")
CRYPTO_BENCHMARKS  = ("BTC-USD", "ETH-USD")
BENCHMARK_PRESETS  = {
    "us":     DEFAULT_BENCHMARKS,
    "korea":  KOREA_BENCHMARKS,
    "crypto": CRYPTO_BENCHMARKS,
}

# ── 데이터 품질 임계값 ───────────────────────────────────────────────────────
# [FIX 2차 A-3] 인라인 매직 넘버 → 이름 있는 상수
MIN_DATA_ROWS                = 30      # 분석에 필요한 최소 거래일 수
DAILY_RETURN_ALERT_THRESHOLD = 0.35   # 일간 변동률 이상치 경계 (35%)
CLOSE_RANGE_ALERT_RATIO      = 4.0    # 기간 최고/최저 비율 경계 (4배)
LATEST_VS_MEDIAN_HIGH        = 3.0    # 최근 종가/중앙값 상한 (3배)
LATEST_VS_MEDIAN_LOW         = 0.33   # 최근 종가/중앙값 하한 (1/3)
ADJ_CLOSE_DIFF_THRESHOLD     = 0.01   # 수정종가/종가 차이 경계 (1%)
REFERENCE_PRICE_DIFF_PCT     = 1.0    # 외부 기준 가격 차이 경계 (1%)

# ── 시각화 상수 ──────────────────────────────────────────────────────────────
CHART_ROW_HEIGHTS   = (0.34, 0.12, 0.16, 0.16, 0.22)
CHART_MARGIN_LARGE  = 135   # 어노테이션 3줄 이상 (summary + 2개 섹션 이상)
CHART_MARGIN_MED    = 110   # 어노테이션 1~2줄
CHART_MARGIN_NONE   = 90    # 어노테이션 없음
MA_COLORS = {
    "MA5":   "#FFA726",   # 주황
    "MA20":  "#42A5F5",   # 파랑
    "MA60":  "#AB47BC",   # 보라
    "MA120": "#EC407A",   # 핑크 (캔들 상승색 #EF5350 과 혼동 방지)
}
BENCHMARK_LINE_COLORS = ["#00BCD4", "#FFD54F", "#A5D6A7", "#CE93D8"]


# ──────────────────────────────────────────────────────────────────────────────
# 의존성: import 실패를 None 으로 보존 → 실행 시점에만 에러 발생
# (import 시점 sys.exit 제거 → 순수 함수 단위 테스트·부분 import 가능)
# ──────────────────────────────────────────────────────────────────────────────
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


def _check_deps(need_plotly: bool = True,
                need_fdr: bool = False,
                need_yfinance: bool = True) -> None:
    """
    실행 시점에 필수 의존성을 확인한다.
    import 시점이 아닌 실제 사용 직전에 호출해 에러를 늦춤으로써
    계산 함수만 import 해 테스트하는 경우를 허용한다.
    """
    missing: list[str] = []
    if need_yfinance and not _yfinance_ok:
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
    """핵심 파라미터 사전 검증. 문제가 있으면 ValueError 를 발생시킨다."""
    if not ticker or not ticker.strip():
        raise ValueError("ticker 는 빈 문자열일 수 없습니다.")
    if initial_capital <= 0:
        raise ValueError(
            f"initial_capital 은 0 보다 커야 합니다. (입력값: {initial_capital:,})"
        )
    if not (0 <= commission < 1):
        raise ValueError(
            f"commission 은 0 이상 1 미만이어야 합니다. (입력값: {commission})"
        )
    if period_years <= 0:
        raise ValueError(
            f"period_years 는 0 보다 커야 합니다. (입력값: {period_years})"
        )


def _detect_annualization_days(df: pd.DataFrame) -> int:
    """
    데이터에 주말 행이 있으면 365, 없으면 252 를 반환한다.

    - 주식(평일 거래): 252
    - 암호화폐·24/7 자산: 365
    """
    sample = df.index[:min(60, len(df))]
    return 365 if any(d.weekday() >= 5 for d in sample) else 252


# ──────────────────────────────────────────────
# 벤치마크 설정 헬퍼
# [FIX 1차 H-2 / 2차 B-1] 공통 헬퍼 구역에서 분리
# ──────────────────────────────────────────────

def _parse_benchmarks(value: str) -> tuple[str, ...]:
    """쉼표 구분 벤치마크 티커 파싱 (순서 보존, 중복 제거)."""
    items: list[str] = []
    for item in value.split(","):
        benchmark = item.strip().upper()
        if benchmark and benchmark not in items:
            items.append(benchmark)
    if not items:
        raise argparse.ArgumentTypeError("benchmarks must include at least one ticker")
    return tuple(items)


def _positive_int(value: str) -> int:
    """양의 정수 CLI 값 파싱."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("value must be an integer")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _default_benchmarks_for_ticker(ticker: str) -> tuple[str, ...]:
    """자산 유형에 따른 기본 벤치마크 선택."""
    normalized = ticker.strip().upper()
    if is_korean_ticker(normalized):
        return KOREA_BENCHMARKS
    if normalized.endswith("-USD"):
        filtered = tuple(item for item in CRYPTO_BENCHMARKS if item != normalized)
        return filtered if filtered else DEFAULT_BENCHMARKS
    return DEFAULT_BENCHMARKS


def _select_benchmarks(ticker: str,
                       benchmarks: tuple[str, ...] | None = None,
                       preset: str = "auto") -> tuple[str, ...]:
    """
    명시적 벤치마크 또는 자산 유형 프리셋으로 벤치마크를 결정한다.

    [FIX 1차 B-3] BENCHMARK_PRESETS[preset] KeyError → .get() 방어 처리
    """
    if benchmarks:
        return benchmarks
    if preset == "off":
        return ()
    if preset == "auto":
        selected = _default_benchmarks_for_ticker(ticker)
    else:
        selected = BENCHMARK_PRESETS.get(preset)
        if selected is None:
            raise ValueError(
                f"알 수 없는 benchmark_preset: '{preset}'. "
                f"허용값: {list(BENCHMARK_PRESETS) + ['auto', 'off']}"
            )

    normalized = ticker.strip().upper()
    return tuple(item for item in selected if item != normalized)


def _resolve_output_dir(output_dir: str | Path) -> Path:
    """경로 순회(".." 컴포넌트) 포함 시 ValueError 발생 후 절대 경로 반환."""
    resolved = Path(output_dir).resolve()
    if ".." in Path(output_dir).parts:
        raise ValueError(f"output_dir 에 경로 순회 문자('..')가 포함돼 있습니다: {output_dir!r}")
    return resolved


def _make_html_path(ticker: str,
                    overwrite: bool = False,
                    output_dir: str | Path = ".") -> Path:
    """
    대시보드 HTML 저장 경로를 결정한다.

    overwrite=True  → <ticker>_dashboard.html  (기존 파일 덮어쓰기)
    overwrite=False → <ticker>_YYYYMMDD_HHMMSS_dashboard.html (타임스탬프)

    [FIX 2차 C-4] ".." 등 경로 순회 문자 차단 — 영숫자·하이픈·밑줄만 허용
    [FIX 2차 B-2] output_dir 파라미터 추가 (기본: 현재 디렉터리)
    """
    safe = re.sub(r"[^\w\-]", "_", ticker)   # [FIX] 안전한 파일명
    if overwrite:
        fname = f"{safe}_dashboard.html"
    else:
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{safe}_{ts}_dashboard.html"
    out = _resolve_output_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / fname


# ──────────────────────────────────────────────
# 1단계. 데이터 수집
# ──────────────────────────────────────────────

def is_korean_ticker(ticker: str) -> bool:
    """6자리 숫자이면 한국 종목으로 판단."""
    return ticker.isdigit() and len(ticker) == 6


def fetch_data(ticker: str, period_years: float = DEFAULT_PERIOD_YEARS,
               start: str = None, end: str = None,
               debug_source: bool = False,
               debug_columns_dir: str | Path | None = None) -> pd.DataFrame:
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
    # [FIX 2차 B-3] datetime.today() → UTC 기준 (로컬 시간대 의존 제거)
    now_utc = datetime.now(timezone.utc)
    if end is None:
        end = now_utc.strftime("%Y-%m-%d")
    if start is None:
        start = (now_utc - timedelta(days=int(period_years * 365))).strftime("%Y-%m-%d")

    # 날짜 유효성 검사
    try:
        s_dt = datetime.strptime(start, "%Y-%m-%d")
        e_dt = datetime.strptime(end,   "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"날짜 형식 오류: start={start}, end={end}. 형식: YYYY-MM-DD")

    if s_dt >= e_dt:
        raise ValueError(f"시작일({start})이 종료일({end})보다 같거나 늦습니다.")

    ticker = ticker.strip()
    # 글로벌 종목은 대문자 강제 (yfinance 는 소문자 티커 실패)
    if not is_korean_ticker(ticker):
        ticker = ticker.upper()

    if is_korean_ticker(ticker):
        if fdr is None:
            raise ImportError("FinanceDataReader 미설치. pip install finance-datareader")
        df = _fetch_korean(ticker, start, end)
    else:
        df = _fetch_global(ticker, start, end,
                           debug_source=debug_source,
                           debug_columns_dir=debug_columns_dir)

    if df is None or df.empty:
        raise ValueError(f"'{ticker}' 에 대한 데이터를 가져올 수 없습니다. 티커를 확인하세요.")

    # 컬럼 정규화
    df = _normalize_columns(df)

    # DatetimeIndex 보장
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # 결측값 처리: forward fill 전 현황 경고
    nan_count = df[["Open", "High", "Low", "Close"]].isna().sum().sum()
    if nan_count > 0:
        print(f"[경고] {nan_count}개 결측값 감지 → forward fill 적용")
    df = df.ffill().dropna()

    # [FIX 2차 A-3] 30 → MIN_DATA_ROWS 상수 사용
    if len(df) < MIN_DATA_ROWS:
        raise ValueError(
            f"'{ticker}' 데이터가 너무 짧습니다 ({len(df)}일). 기간을 늘려보세요."
        )

    print(f"[데이터 수집 완료] {ticker} | {df.index[0].date()} ~ {df.index[-1].date()} | {len(df)}거래일")
    return df


def _fetch_korean(ticker: str, start: str, end: str) -> pd.DataFrame:
    """FinanceDataReader 로 한국 주식 수집."""
    try:
        df = fdr.DataReader(ticker, start, end)
        return df
    except Exception as e:
        raise ValueError(f"한국 주식 데이터 수집 실패 ({ticker}): {e}")


def _format_columns_sample(columns, limit: int = 6) -> str:
    """디버그 로그용 컬럼 샘플 문자열."""
    return ", ".join(str(col) for col in list(columns)[:limit])


def _save_debug_columns(ticker: str,
                        columns,
                        output_dir: str | Path) -> Path:
    """Save full raw source columns for yfinance debugging."""
    safe_ticker = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker).strip("_") or "ticker"
    out_dir = _resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_ticker}_yfinance_columns.txt"
    lines = [str(col) for col in list(columns)]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _fetch_global(ticker: str,
                  start: str,
                  end: str,
                  debug_source: bool = False,
                  debug_columns_dir: str | Path | None = None) -> pd.DataFrame:
    """
    yfinance 로 글로벌 주식 수집.

    [FIX 2차 C-1] timeout=30 추가 (무한 대기 방지)
    """
    try:
        raw = yf.download(
            ticker, start=start, end=end,
            auto_adjust=False, progress=False, timeout=30,
        )
        if raw.empty:
            raise ValueError(f"yfinance: '{ticker}' 데이터 없음")

        if debug_source:
            column_type = type(raw.columns).__name__
            print(f"[데이터 원천] {ticker} yfinance columns={column_type} "
                  f"sample=[{_format_columns_sample(raw.columns)}]")

        if debug_columns_dir is not None:
            out_path = _save_debug_columns(ticker, raw.columns, debug_columns_dir)
            print(f"[데이터 원천] {ticker} full raw columns saved: {out_path}")

        raw = _flatten_yfinance_columns(raw, ticker)
        raw = raw.loc[:, ~raw.columns.duplicated(keep="first")]

        if debug_source:
            print(f"[데이터 원천] {ticker} normalized columns="
                  f"[{_format_columns_sample(raw.columns)}]")

        return raw
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"글로벌 주식 데이터 수집 실패 ({ticker}): {e}")


def _flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance 단일 티커 MultiIndex 컬럼에서 OHLCV 레벨을 선택한다."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    ohlcv_names = {"open", "high", "low", "close", "adj close", "volume"}
    for level in range(df.columns.nlevels):
        values = {str(value).lower().strip() for value in df.columns.get_level_values(level)}
        if len(values & ohlcv_names) >= 4:
            flattened = df.copy()
            flattened.columns = flattened.columns.get_level_values(level)
            return flattened

    # 예상과 다른 MultiIndex 구조면 기존 동작에 가깝게 첫 레벨을 사용한다.
    flattened = df.copy()
    flattened.columns = flattened.columns.get_level_values(0)
    print(f"[경고] yfinance MultiIndex 컬럼 구조 확인 필요 ({ticker}): {list(df.columns)[:6]}")
    return flattened


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼명을 표준 형식으로 통일.

    [FIX 2차 B-4] 동일 표준 이름으로 여러 컬럼이 매핑될 때 중복 감지·경고
    """
    # 정확한 이름만 허용 — 부분 문자열 매칭은 "inflow"→Low 같은 오매핑을 유발할 수 있음
    _exact: dict[str, str] = {
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
        "vol":    "Volume",
    }

    rename_map: dict = {}
    seen_targets: set[str] = set()

    for col in df.columns:
        col_lower = str(col).lower().strip().replace(" ", "_")
        # 'adj' + 'close' 를 먼저 체크 (수정종가 변형명이 다양해 부분 문자열 방식 유지)
        if "adj" in col_lower and "close" in col_lower:
            target = "Adj Close"
        elif col_lower in _exact:
            target = _exact[col_lower]
        else:
            continue

        if target in seen_targets:
            print(f"[경고] 컬럼 '{col}' → '{target}' 중복 매핑 감지. 첫 번째 컬럼을 사용합니다.")
            continue
        seen_targets.add(target)
        rename_map[col] = target

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


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 품질 검사
# [FIX 2차 A-4] 계산 로직과 콘솔 출력 분리
#   _compute_data_quality_stats   → 순수 계산 (부작용 없음, 단위 테스트 가능)
#   print_data_quality_check      → 출력 래퍼 (계산 함수 호출 후 결과 출력)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_data_quality_stats(ticker: str, df: pd.DataFrame) -> dict:
    """한국 종목 데이터 품질 지표 계산 (부작용 없음)."""
    if not is_korean_ticker(ticker):
        return {"checked": False, "reason": "not_korean_ticker"}

    latest      = df.iloc[-1]
    close       = float(latest["Close"])
    adj_close   = float(latest["Adj Close"])
    ratio       = adj_close / close if close else 0.0
    min_close   = float(df["Close"].min())
    max_close   = float(df["Close"].max())
    median_close = float(df["Close"].median())
    close_range_ratio  = max_close / min_close if min_close else 0.0
    latest_vs_median   = close / median_close if median_close else 0.0
    max_abs_daily_return = float(df["Close"].pct_change().abs().max(skipna=True) or 0.0)

    # [FIX 1차 B-1] 지역 변수명 'warnings' → 'warning_codes' (stdlib warnings 섀도잉 방지)
    warning_codes: list[str] = []
    if abs(ratio - 1.0) > ADJ_CLOSE_DIFF_THRESHOLD:
        warning_codes.append("adjusted_close_diff")
    if close_range_ratio >= CLOSE_RANGE_ALERT_RATIO:
        warning_codes.append("wide_close_range")
    if latest_vs_median >= LATEST_VS_MEDIAN_HIGH or latest_vs_median <= LATEST_VS_MEDIAN_LOW:
        warning_codes.append("latest_close_outlier")
    if max_abs_daily_return > DAILY_RETURN_ALERT_THRESHOLD:
        warning_codes.append("large_daily_move")

    return {
        "checked":              True,
        "ticker":               ticker,
        "latest_close":         close,
        "latest_adj_close":     adj_close,
        "adj_close_ratio":      ratio,
        "min_close":            min_close,
        "max_close":            max_close,
        "close_range_ratio":    close_range_ratio,
        "latest_vs_median":     latest_vs_median,
        "max_abs_daily_return": max_abs_daily_return,
        "warning":              warning_codes[0] if warning_codes else None,
        "warnings":             warning_codes,
    }


def print_data_quality_check(ticker: str, df: pd.DataFrame) -> dict:
    """데이터 품질 검사 결과를 콘솔에 출력하고 결과 dict 를 반환."""
    result = _compute_data_quality_stats(ticker, df)
    if not result["checked"]:
        return result

    r = result
    print(f"[데이터 검증] {r['ticker']} | 최근 종가 {r['latest_close']:,.0f} | "
          f"수정종가/종가 {r['adj_close_ratio']:.3f} | "
          f"기간 종가 범위 {r['min_close']:,.0f}~{r['max_close']:,.0f}")

    for code in r["warnings"]:
        if code == "adjusted_close_diff":
            print("[경고] 수정종가와 종가 차이가 큽니다. 액면분할/배당 조정 여부를 확인하세요.")
        elif code == "wide_close_range":
            print(f"[경고] 기간 중 종가 범위가 넓습니다 ({r['close_range_ratio']:.2f}배). "
                  "액면분할/거래정지/데이터 원천을 확인하세요.")
        elif code == "latest_close_outlier":
            print(f"[경고] 최근 종가가 기간 중앙값과 크게 다릅니다 ({r['latest_vs_median']:.2f}배). "
                  "최신 원천 데이터를 확인하세요.")
        elif code == "large_daily_move":
            print(f"[경고] 일간 종가 변동률이 비정상적으로 큽니다 "
                  f"({r['max_abs_daily_return']*100:.2f}%). 원천 데이터 이상 여부를 확인하세요.")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 외부 기준 가격 비교
# [FIX 2차 A-4] 계산 로직과 콘솔 출력 분리
# ──────────────────────────────────────────────────────────────────────────────

def _compute_external_price_stats(ticker: str, df: pd.DataFrame) -> dict:
    """한국 종목과 yfinance .KS/.KQ 기준 가격 비교 계산 (부작용 없음)."""
    if not is_korean_ticker(ticker):
        return {"checked": False, "reason": "not_korean_ticker"}
    if not _yfinance_ok:
        return {"checked": False, "reason": "yfinance_not_installed"}

    start = df.index[max(len(df) - 10, 0)].strftime("%Y-%m-%d")  # 최근 10거래일 기준 비교
    end   = (df.index[-1] + timedelta(days=2)).strftime("%Y-%m-%d")  # yfinance end 는 exclusive — 주말·반환 범위 편차 대비 2일 여유
    reference_df     = None
    reference_ticker = None
    failures: dict   = {}

    for suffix in (".KS", ".KQ"):
        candidate = f"{ticker}{suffix}"
        try:
            reference_df = _normalize_columns(
                _fetch_global(candidate, start, end)
            ).ffill().dropna()
            reference_ticker = candidate
            break
        except Exception as exc:
            failures[candidate] = str(exc)

    if reference_df is None or reference_df.empty:
        reason_detail = ("; ".join(f"{k}: {v}" for k, v in failures.items()) or "no data")
        return {
            "checked":       False,
            "reason":        "reference_not_found",
            "reason_detail": reason_detail,
            "failures":      failures,
        }

    local_close = float(df["Close"].iloc[-1])
    ref_close   = float(reference_df["Close"].iloc[-1])
    diff_pct    = (local_close / ref_close - 1) * 100 if ref_close else 0.0
    warning     = "reference_price_diff" if abs(diff_pct) > REFERENCE_PRICE_DIFF_PCT else None

    return {
        "checked":          True,
        "ticker":           ticker,
        "reference_ticker": reference_ticker,
        "local_close":      local_close,
        "reference_close":  ref_close,
        "diff_pct":         diff_pct,
        "warning":          warning,
        "failures":         failures,
    }


def print_external_price_check(ticker: str, df: pd.DataFrame) -> dict:
    """외부 기준 가격 비교 결과를 콘솔에 출력하고 결과 dict 를 반환."""
    result = _compute_external_price_stats(ticker, df)
    if not result["checked"]:
        reason = result.get("reason", "")
        if reason == "yfinance_not_installed":
            print("[외부 기준 가격] yfinance 미설치로 비교를 건너뜁니다.")
        elif reason == "reference_not_found":
            print(f"[외부 기준 가격] yfinance .KS/.KQ 기준 가격을 찾지 못했습니다. "
                  f"사유: {result.get('reason_detail', 'no data')}")
        # not_korean_ticker 는 출력 없음
        return result

    r = result
    print(f"[외부 기준 가격] FDR {r['ticker']} {r['local_close']:,.0f} | "
          f"yfinance {r['reference_ticker']} {r['reference_close']:,.0f} | "
          f"차이 {r['diff_pct']:+.2f}%")
    if r["warning"]:
        print("[경고] 원천 데이터와 외부 기준 가격 차이가 큽니다. 종목 매핑/수정주가를 확인하세요.")
    return result


# ──────────────────────────────────────────────
# 2단계. 수익률 / 리스크 지표 계산
# ──────────────────────────────────────────────

def _print_data_validation_reports(ticker: str, df: pd.DataFrame) -> tuple[dict, dict]:
    """Print data validation reports and return their result dictionaries."""
    data_quality         = print_data_quality_check(ticker, df)
    external_price_check = print_external_price_check(ticker, df)
    return data_quality, external_price_check


def compute_returns(df: pd.DataFrame,
                    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
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
        'daily_return'      : Series,
        'cumulative_return' : Series,
        'annual_volatility' : float,
        'sharpe_ratio'      : float,
        'mdd'               : float,
        'annual_return'     : float,
        'annualization_days': int,
    }
    """
    close = df["Close"]

    # NaN 유지: std 계산에서 인위적 0 수익률 제외
    daily_ret = close.pct_change()

    # 누적 수익률: 첫날 NaN → 0 으로 채워 cumprod 계산
    cum_ret = (1 + daily_ret.fillna(0)).cumprod()

    # 실제 수익률 개수 (첫 NaN 제외) 로 연환산
    n_days    = max(len(daily_ret.dropna()), 1)
    final_cum = cum_ret.iloc[-1]

    if final_cum > 0 and n_days > 0:
        annual_ret = (final_cum ** (annualization_days / n_days)) - 1
    else:
        annual_ret = 0.0

    # NaN 제외한 실제 수익률로 std 계산
    annual_vol = daily_ret.std() * math.sqrt(annualization_days)

    # 극소값 방어 (부동소수점 오차 대비)
    if annual_vol > 1e-8:
        sharpe = (annual_ret - risk_free_rate) / annual_vol
    else:
        sharpe = 0.0

    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    mdd      = drawdown.min()   # 음수값; 리포트에서 *100 하면 음수 퍼센트

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
# 벤치마크 비교
# [FIX 2차 H-2 / B-1] 기술지표 섹션에서 독립 분리
# ──────────────────────────────────────────────

def _fetch_one_benchmark(benchmark: str,
                         start: str,
                         end: str,
                         risk_free_rate: float,
                         annualization_days: int,
                         corr_window: int,
                         target_cum: pd.Series,
                         target_daily: pd.Series,
                         target_total_return: float,
                         debug: bool = False,
                         debug_source: bool = False,
                         debug_columns_dir: str | Path | None = None) -> dict:
    """
    단일 벤치마크 데이터 수집 + 통계 계산 (ThreadPoolExecutor 내 실행용).

    [FIX 1차 B-5] rolling_corr.dropna() 이중 계산 → 단일 변수로 통합
    """
    try:
        if debug_source or debug_columns_dir is not None:
            bm_df = fetch_data(
                benchmark, start=start, end=end,
                debug_source=debug_source,
                debug_columns_dir=debug_columns_dir,
            )
        else:
            bm_df = fetch_data(benchmark, start=start, end=end)
        first_close = float(bm_df["Close"].iloc[0])
        last_close  = float(bm_df["Close"].iloc[-1])
        if debug:
            print(f"[벤치마크 검증] {benchmark} | "
                  f"{bm_df.index[0].date()} {first_close:,.2f} → "
                  f"{bm_df.index[-1].date()} {last_close:,.2f} | {len(bm_df)}행")

        bm_stats = compute_returns(bm_df,
                                   risk_free_rate=risk_free_rate,
                                   annualization_days=annualization_days)
        total_return = float(bm_stats["cumulative_return"].iloc[-1] - 1)

        aligned_cum = pd.concat(
            [target_cum.rename("target"),
             bm_stats["cumulative_return"].rename("benchmark")],
            axis=1,
            sort=False,
        ).ffill().dropna()
        excess_return_series = aligned_cum["target"] - aligned_cum["benchmark"]

        aligned_daily = pd.concat(
            [target_daily.rename("target"),
             bm_stats["daily_return"].rename("benchmark")],
            axis=1,
            sort=False,
        ).dropna()
        rolling_corr = aligned_daily["target"].rolling(corr_window).corr(
            aligned_daily["benchmark"]
        )
        # [FIX 1차 B-5] dropna() 한 번만 호출
        _corr_valid  = rolling_corr.dropna()
        latest_corr  = None if _corr_valid.empty else float(_corr_valid.iloc[-1])

        return {
            "ticker":               benchmark,
            "total_return":         total_return,
            "annual_return":        float(bm_stats["annual_return"]),
            "mdd":                  float(bm_stats["mdd"]),
            "excess_return":        target_total_return - total_return,
            "cumulative_return":    bm_stats["cumulative_return"],
            "excess_return_series": excess_return_series,
            "rolling_corr":         rolling_corr,
            "corr_window":          corr_window,
            "latest_corr":          latest_corr,
            "first_close":          first_close,
            "last_close":           last_close,
            "n_rows":               len(bm_df),
            "error":                None,
        }
    except Exception as exc:
        return {
            "ticker":               benchmark,
            "total_return":         None,
            "annual_return":        None,
            "mdd":                  None,
            "excess_return":        None,
            "cumulative_return":    None,
            "excess_return_series": None,
            "rolling_corr":         None,
            "corr_window":          corr_window,
            "latest_corr":          None,
            "first_close":          None,
            "last_close":           None,
            "n_rows":               None,
            "error":                str(exc),
        }


def compute_benchmark_comparison(target_stats: dict,
                                  start: str,
                                  end: str,
                                  risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
                                  annualization_days: int = 252,
                                  benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS,
                                  corr_window: int = 60,
                                  debug: bool = False,
                                  debug_source: bool = False,
                                  debug_columns_dir: str | Path | None = None) -> list[dict]:
    """
    대상 종목 수익률 지표를 시장 벤치마크와 비교한다.

    벤치마크 수집 실패는 error 필드로 기록하며 전체 분석을 중단하지 않는다.

    벤치마크는 yfinance 동시 호출 중복 가능성을 피하기 위해 순차 수집한다.
    """
    if not benchmarks:
        return []

    target_total_return = float(target_stats["cumulative_return"].iloc[-1] - 1)
    target_cum          = target_stats["cumulative_return"]
    target_daily        = target_stats["daily_return"]

    # 공통 kwargs
    kwargs = dict(
        start=start,
        end=end,
        risk_free_rate=risk_free_rate,
        annualization_days=annualization_days,
        corr_window=corr_window,
        target_cum=target_cum,
        target_daily=target_daily,
        target_total_return=target_total_return,
        debug=debug,
        debug_source=debug_source,
        debug_columns_dir=debug_columns_dir,
    )

    rows = [_fetch_one_benchmark(bm, **kwargs) for bm in benchmarks]
    if debug:
        seen: dict[tuple, str] = {}
        for row in rows:
            if row.get("error"):
                continue
            signature = (row.get("n_rows"), row.get("first_close"), row.get("last_close"))
            if signature in seen:
                print(f"[벤치마크 경고] {seen[signature]}와 {row['ticker']}의 "
                      "행 수/첫 종가/마지막 종가가 같습니다. 원천 데이터를 확인하세요.")
            else:
                seen[signature] = row["ticker"]

    return rows


def print_benchmark_report(ticker: str,
                            target_stats: dict,
                            benchmark_rows: list[dict]) -> None:
    """벤치마크 비교 테이블을 콘솔에 출력."""
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
        corr_text = "N/A" if row["latest_corr"] is None else f"{row['latest_corr']:.2f}"
        print(f"    {row['corr_window']}D rolling correlation: {corr_text}")


# ──────────────────────────────────────────────
# 3단계. 기술적 지표
# ──────────────────────────────────────────────

def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """5/20/60/120일 이동평균선 및 골든/데드 크로스 시그널 추가.

    입력 df 를 직접 수정(in-place)하고 반환한다. 원본을 보호하려면 add_indicators() 를 사용하라.
    """
    close = df["Close"]
    df["MA5"]   = close.rolling(window=5,   min_periods=5).mean()
    df["MA20"]  = close.rolling(window=20,  min_periods=20).mean()
    df["MA60"]  = close.rolling(window=60,  min_periods=60).mean()
    df["MA120"] = close.rolling(window=120, min_periods=120).mean()

    # NaN 구간에서는 비교가 False → 시그널 없음 (의도된 동작)
    ma20_prev   = df["MA20"].shift(1)
    ma60_prev   = df["MA60"].shift(1)
    cond_golden = (df["MA20"] > df["MA60"]) & (ma20_prev <= ma60_prev)
    cond_dead   = (df["MA20"] < df["MA60"]) & (ma20_prev >= ma60_prev)
    df["MA_Signal"] = np.where(cond_golden, "golden", np.where(cond_dead, "dead", ""))
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI 계산 (Wilder's smoothing) 및 buy/sell/neutral 시그널 추가.

    Wilder's RSI: alpha = 1/period  (ewm com = period - 1)

    세 가지 케이스:
      avg_gain > 0, avg_loss = 0 → RSI = 100 (상승만 존재)
      avg_gain = 0, avg_loss = 0 → RSI = 50  (flat, 중립)
      avg_gain = 0, avg_loss > 0 → RSI = 0   (하락만 존재, 정상 공식)
      NaN (워밍업 미완료)         → RSI = NaN

    입력 df 를 직접 수정(in-place)하고 반환한다. 원본을 보호하려면 add_indicators() 를 사용하라.
    """
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        both_zero = (avg_gain == 0) & (avg_loss == 0)
        rs = np.where(avg_loss == 0, np.inf, avg_gain.values / avg_loss.values)

    rsi_vals = np.where(
        both_zero,
        50.0,
        np.where(np.isinf(rs), 100.0, 100.0 - (100.0 / (1.0 + rs)))
    )
    rsi = pd.Series(rsi_vals, index=df.index)
    # 워밍업 구간 (avg_gain/avg_loss 가 NaN) → RSI 도 NaN 유지
    rsi = rsi.where(avg_gain.notna() & avg_loss.notna(), other=np.nan)

    df["RSI"] = rsi
    # RSI NaN 구간은 "" 처리 (기존 "neutral" 오표기 방지)
    df["RSI_Signal"] = np.where(
        rsi.isna(), "",
        np.where(rsi >= 70, "sell",
                 np.where(rsi <= 30, "buy", "neutral"))
    )
    return df


def compute_macd(df: pd.DataFrame,
                 fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD Line, Signal Line, Histogram 및 크로스 시그널 추가.

    Note: adjust=False EMA 특성상 초기(slow 기간 이전) 값은 수렴이 덜 됨.
          워밍업 구간에 대한 별도 NaN 처리는 수행하지 않음 (거래 시그널에는
          min_periods 가 없어 미미한 영향이지만 극초기 구간 해석 주의).

    입력 df 를 직접 수정(in-place)하고 반환한다. 원본을 보호하려면 add_indicators() 를 사용하라.
    """
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
    """모든 기술적 지표를 DataFrame 에 추가. 원본 복사 후 반환."""
    # copy(): 외부에서 전달된 df 원본 보호 (compute_* 함수들이 인플레이스 대입)
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
    close  = float(latest["Close"])
    ma20   = latest.get("MA20",        np.nan)
    ma60   = latest.get("MA60",        np.nan)
    ma120  = latest.get("MA120",       np.nan)
    rsi    = latest.get("RSI",         np.nan)
    macd   = latest.get("MACD",        np.nan)
    sig    = latest.get("MACD_Signal", np.nan)

    price_position = {
        "above_ma20":  bool(pd.notna(ma20)  and close > ma20),
        "above_ma60":  bool(pd.notna(ma60)  and close > ma60),
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

    if pd.isna(macd) or pd.isna(sig):
        macd_state = "판단 보류"
    elif macd > sig:
        macd_state = "상승 모멘텀"
    else:
        macd_state = "하락/둔화 모멘텀"

    score_checks = [
        price_position["above_ma20"],
        price_position["above_ma60"],
        bool(pd.notna(ma20) and pd.notna(ma60) and ma20 > ma60),
        bool(pd.notna(macd) and pd.notna(sig)  and macd > sig),
        bool(pd.notna(rsi)  and 40 <= rsi <= 70),
    ]
    signal_score = sum(score_checks)

    return {
        "date":             df.index[-1],
        "close":            close,
        "trend":            trend,
        "price_position":   price_position,
        "rsi":              None if pd.isna(rsi)  else float(rsi),
        "rsi_state":        rsi_state,
        "macd":             None if pd.isna(macd) else float(macd),
        "macd_signal":      None if pd.isna(sig)  else float(sig),
        "macd_state":       macd_state,
        "signal_score":     signal_score,
        "signal_score_max": len(score_checks),
    }


def print_current_state(summary: dict, ticker: str) -> None:
    """현재 기술적 상태 요약을 콘솔에 출력."""
    date = (summary["date"].strftime("%Y-%m-%d")
            if hasattr(summary["date"], "strftime")
            else str(summary["date"]))
    price    = summary["price_position"]
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
    """대시보드 헤더용 현재 상태 요약 HTML 문자열."""
    rsi      = summary.get("rsi")
    rsi_text = "N/A" if (rsi is None or pd.isna(rsi)) else f"{rsi:.1f}"
    return (
        f"<b>현재 상태</b>: {summary.get('trend', 'N/A')} | "
        f"RSI {rsi_text} ({summary.get('rsi_state', 'N/A')}) | "
        f"MACD {summary.get('macd_state', 'N/A')} | "
        f"점수 {summary.get('signal_score', 0)}/{summary.get('signal_score_max', 0)}"
    )


def _format_benchmark_annotation(benchmark_rows: list[dict] | None) -> str | None:
    """대시보드 헤더용 벤치마크 비교 요약 HTML 문자열."""
    if not benchmark_rows:
        return None
    parts = []
    for row in benchmark_rows:
        if row.get("error"):
            parts.append(f"{row['ticker']} N/A")
        else:
            corr       = row.get("latest_corr")
            corr_text  = "N/A" if corr is None else f"{corr:.2f}"
            corr_window = row.get("corr_window", 60)
            parts.append(
                f"{row['ticker']} 초과 {row['excess_return']*100:+.2f}% "
                f"/ {corr_window}D상관 {corr_text}"
            )
    return "<b>벤치마크</b>: " + " | ".join(parts)


def _format_strategy_annotation(strategy_summary: dict | None) -> str | None:
    """대시보드 헤더용 백테스트 우위 전략 요약 HTML 문자열."""
    if not strategy_summary:
        return None
    return (
        f"<b>전략</b>: 우위 {strategy_summary['winner']} | "
        f"수익률 차이 {strategy_summary['return_diff']*100:+.2f}% | "
        f"샤프 차이 {strategy_summary['sharpe_diff']:+.2f}"
    )


def _format_data_warning_annotation(data_quality: dict | None) -> str | None:
    """대시보드 헤더용 데이터 품질 경고 요약 HTML 문자열."""
    if not data_quality or not data_quality.get("checked"):
        return None
    warnings = data_quality.get("warnings") or []
    if not warnings:
        return None
    return (
        "<span style='color:#FFB74D'><b>데이터 주의</b>: "
        + ", ".join(warnings)
        + "</span>"
    )


def _format_dashboard_summary_annotation(current_state: dict | None,
                                         stats: dict,
                                         benchmark_rows: list[dict] | None,
                                         strategy_summary: dict | None,
                                         data_quality: dict | None = None) -> str:
    """상단 대시보드 요약을 현재 상태·성과·벤치마크/전략 줄로 구분한다."""
    parts = []
    if current_state:
        parts.append(_format_current_state_annotation(current_state))

    parts.append(
        f"<b>성과</b>: 연수익률 {stats['annual_return']*100:+.2f}% | "
        f"연변동성 {stats['annual_volatility']*100:.2f}% | "
        f"Sharpe {stats['sharpe_ratio']:.2f} | "
        f"MDD {stats['mdd']*100:.2f}%"
    )

    benchmark_annotation = _format_benchmark_annotation(benchmark_rows)
    if benchmark_annotation:
        parts.append(benchmark_annotation)

    strategy_annotation = _format_strategy_annotation(strategy_summary)
    if strategy_annotation:
        parts.append(strategy_annotation)

    data_warning_annotation = _format_data_warning_annotation(data_quality)
    if data_warning_annotation:
        parts.append(data_warning_annotation)

    return "<br>".join(parts)


# ── 차트 패널 빌더 (내부 헬퍼)
# [FIX 2차 A-2] plot_dashboard 249줄 단일 함수 → 패널별 분리

def _add_candle_panel(fig,
                      df: pd.DataFrame,
                      show_marker_legend: bool,
                      show_signal_markers: bool) -> None:
    """차트1: 캔들스틱 + 이동평균선 + 골든/데드 크로스 마커."""
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="캔들",
        legendgroup="price",
        legendgrouptitle_text="가격",
        increasing_line_color="#EF5350",
        increasing_fillcolor="rgba(239,83,80,0.85)",
        decreasing_line_color="#26A69A",
        decreasing_fillcolor="rgba(38,166,154,0.85)",
    ), row=1, col=1)

    # [FIX 2차 A-3] MA_COLORS 모듈 상수 사용
    for ma, color in MA_COLORS.items():
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ma], name=ma,
                line=dict(color=color, width=1.2), opacity=0.85,
                legendgroup="price",
            ), row=1, col=1)

    if show_signal_markers:
        golden = df[(df["MA_Signal"] == "golden") & df["MA20"].notna()]
        dead   = df[(df["MA_Signal"] == "dead")   & df["MA20"].notna()]
        if not golden.empty:
            fig.add_trace(go.Scatter(
                x=golden.index, y=golden["MA20"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="#00E676"),
                name="골든크로스", showlegend=show_marker_legend,
                legendgroup="price",
            ), row=1, col=1)
        if not dead.empty:
            fig.add_trace(go.Scatter(
                x=dead.index, y=dead["MA20"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="#FF1744"),
                name="데드크로스", showlegend=show_marker_legend,
                legendgroup="price",
            ), row=1, col=1)


def _add_volume_panel(fig, df: pd.DataFrame) -> None:
    """차트2: 거래량 바.

    [FIX 2차 C-3] Python 리스트 컴프리헨션 → np.where 벡터화
    """
    vol_colors = np.where(
        df["Close"].values >= df["Open"].values, "#EF5350", "#26A69A"
    ).tolist()
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="거래량",
        marker_color=vol_colors, opacity=0.7,
        legendgroup="volume",
        legendgrouptitle_text="거래량",
    ), row=2, col=1)


def _add_rsi_panel(fig,
                   df: pd.DataFrame,
                   show_marker_legend: bool,
                   show_signal_markers: bool) -> None:
    """차트3: RSI + 기준선 + 시그널 마커."""
    rsi_valid = df["RSI"].dropna()
    fig.add_trace(go.Scatter(
        x=rsi_valid.index, y=rsi_valid, name="RSI",
        line=dict(color="#FF9800", width=1.5),
        legendgroup="indicator",
        legendgrouptitle_text="지표",
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",  opacity=0.6, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="blue", opacity=0.6, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot",  line_color="gray", opacity=0.4, row=3, col=1)

    if show_signal_markers:
        rsi_buy  = df[(df["RSI_Signal"] == "buy")  & df["RSI"].notna()]
        rsi_sell = df[(df["RSI_Signal"] == "sell") & df["RSI"].notna()]
        if not rsi_buy.empty:
            fig.add_trace(go.Scatter(
                x=rsi_buy.index, y=rsi_buy["RSI"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#00BCD4"),
                name="RSI 매수", showlegend=show_marker_legend,
                legendgroup="indicator",
            ), row=3, col=1)
        if not rsi_sell.empty:
            fig.add_trace(go.Scatter(
                x=rsi_sell.index, y=rsi_sell["RSI"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#E91E63"),
                name="RSI 매도", showlegend=show_marker_legend,
                legendgroup="indicator",
            ), row=3, col=1)


def _add_macd_panel(fig,
                    df: pd.DataFrame,
                    show_marker_legend: bool,
                    show_signal_markers: bool) -> None:
    """차트4: MACD + Signal + 히스토그램 + 크로스 마커.

    [FIX 2차 C-3] 히스토그램 색상 리스트 컴프리헨션 → np.where 벡터화
    """
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], name="MACD",
        line=dict(color="#2196F3", width=1.5),
        legendgroup="indicator",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"], name="Signal",
        line=dict(color="#FF5722", width=1.5),
        legendgroup="indicator",
    ), row=4, col=1)

    hist_colors = np.where(
        df["MACD_Hist"].fillna(0).values >= 0, "#EF5350", "#26A69A"
    ).tolist()
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"], name="히스토그램",
        marker_color=hist_colors, opacity=0.6,
        legendgroup="indicator",
    ), row=4, col=1)

    if show_signal_markers:
        macd_golden = df[(df["MACD_Cross"] == "golden") & df["MACD"].notna()]
        macd_dead   = df[(df["MACD_Cross"] == "dead")   & df["MACD"].notna()]
        if not macd_golden.empty:
            fig.add_trace(go.Scatter(
                x=macd_golden.index, y=macd_golden["MACD"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#00E676"),
                name="MACD 골든", showlegend=show_marker_legend,
                legendgroup="indicator",
            ), row=4, col=1)
        if not macd_dead.empty:
            fig.add_trace(go.Scatter(
                x=macd_dead.index, y=macd_dead["MACD"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#FF1744"),
                name="MACD 데드", showlegend=show_marker_legend,
                legendgroup="indicator",
            ), row=4, col=1)


def _add_cum_return_panel(fig,
                           df: pd.DataFrame,
                           ticker: str,
                           stats: dict,
                           benchmark_comparison: list[dict] | None,
                           show_cumulative_lines: bool,
                           show_excess_lines: bool,
                           show_benchmark_legend: bool) -> None:
    """차트5: 대상 + 벤치마크 누적수익률 비교."""
    target_cum_pct = (stats["cumulative_return"] - 1) * 100

    if show_cumulative_lines:
        fig.add_trace(go.Scatter(
            x=target_cum_pct.index, y=target_cum_pct,
            name=f"{ticker} 누적수익률",
            line=dict(color="#FFFFFF", width=2.0),
            showlegend=show_benchmark_legend,
            legendgroup="benchmark",
            legendgrouptitle_text="벤치마크",
        ), row=5, col=1)

    for idx, row in enumerate(benchmark_comparison or []):
        series = row.get("cumulative_return")
        if row.get("error") or series is None:
            continue
        color = BENCHMARK_LINE_COLORS[idx % len(BENCHMARK_LINE_COLORS)]

        if show_cumulative_lines:
            fig.add_trace(go.Scatter(
                x=series.index,
                y=(series - 1) * 100,
                name=f"{row['ticker']} 누적수익률",
                line=dict(color=color, width=1.5, dash="dot"),
                showlegend=show_benchmark_legend,
                legendgroup="benchmark",
            ), row=5, col=1)

        excess_series = row.get("excess_return_series")
        if show_excess_lines and excess_series is not None:
            fig.add_trace(go.Scatter(
                x=excess_series.index,
                y=excess_series * 100,
                name=f"{ticker} vs {row['ticker']} 초과수익",
                line=dict(color=color, width=1.2, dash="dash"),
                opacity=0.75,
                showlegend=show_benchmark_legend,
                legendgroup="benchmark",
            ), row=5, col=1)


def plot_dashboard(df: pd.DataFrame,
                   ticker: str,
                   stats: dict,
                   current_state: dict | None = None,
                   benchmark_comparison: list[dict] | None = None,
                   show_excess_return: bool = True,
                   benchmark_display: str = "cumulative",
                   show_marker_legend: bool = False,
                   show_signal_markers: bool = False,
                   strategy_summary: dict | None = None,
                   data_quality: dict | None = None,
                   show_benchmark_legend: bool = False):
    """
    Plotly 서브플롯 대시보드 (캔들+MA, 거래량, RSI, MACD, 누적수익률).

    [FIX 2차 A-2] 249줄 단일 함수 → _add_*_panel 헬퍼로 분리
    """
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        row_heights=list(CHART_ROW_HEIGHTS),
        vertical_spacing=0.03,
        subplot_titles=[
            f"{ticker} 캔들스틱 + 이동평균선",
            "거래량",
            "RSI (14)",
            "MACD (12/26/9)",
            "누적수익률 비교",
        ],
    )

    _add_candle_panel(fig, df, show_marker_legend, show_signal_markers)
    _add_volume_panel(fig, df)
    _add_rsi_panel(fig, df, show_marker_legend, show_signal_markers)
    _add_macd_panel(fig, df, show_marker_legend, show_signal_markers)

    show_cumulative_lines = benchmark_display in ("all", "cumulative")
    show_excess_lines     = show_excess_return and benchmark_display in ("all", "excess")
    _add_cum_return_panel(fig, df, ticker, stats, benchmark_comparison,
                          show_cumulative_lines, show_excess_lines,
                          show_benchmark_legend)

    # ── 레이아웃 ──
    annual_ret_pct = stats["annual_return"]    * 100
    annual_vol_pct = stats["annual_volatility"] * 100
    mdd_pct        = stats["mdd"]              * 100

    dashboard_summary = _format_dashboard_summary_annotation(
        current_state, stats, benchmark_comparison, strategy_summary, data_quality
    )
    # 어노테이션 줄 수(<br> 개수 + 1)에 따라 margin 결정
    if not dashboard_summary:
        top_margin = CHART_MARGIN_NONE
    elif dashboard_summary.count("<br>") >= 2:
        top_margin = CHART_MARGIN_LARGE
    else:
        top_margin = CHART_MARGIN_MED

    fig.update_layout(
        title=None,
        height=1120,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.005,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=top_margin, b=40),
    )

    fig.update_yaxes(title_text="주가",        row=1, col=1)
    fig.update_yaxes(title_text="거래량",       row=2, col=1)
    fig.update_yaxes(title_text="RSI",          row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",         row=4, col=1)
    fig.update_yaxes(title_text="누적수익률(%)", row=5, col=1)

    # 현재 상태·성과·벤치마크·전략 요약 어노테이션
    if dashboard_summary:
        fig.add_annotation(
            text=dashboard_summary,
            xref="paper", yref="paper", x=0, y=1.035,
            xanchor="left", yanchor="bottom", showarrow=False, align="left",
            font=dict(size=12.5, color="#F5F5F5"),
            bgcolor="rgba(20,24,31,0.78)",
            bordercolor="rgba(255,255,255,0.22)",
            borderwidth=1, borderpad=7,
        )

    # 주식만 주말 갭 제거 (암호화폐·24시간 자산은 주말 데이터 존재 → 적용 제외)
    has_weekend = any(d.weekday() >= 5 for d in df.index[:min(60, len(df))])
    if not has_weekend:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    return fig


def _save_and_show_dashboard(fig,
                             ticker: str,
                             overwrite_html: bool,
                             output_dir: str | Path) -> Path:
    """Save the Plotly dashboard HTML, show it, and return the saved path."""
    out_path = _make_html_path(ticker,
                               overwrite=overwrite_html,
                               output_dir=output_dir)
    fig.write_html(out_path)
    print(f"[차트 저장] {out_path}")
    fig.show()
    return out_path


# ──────────────────────────────────────────────
# 5단계. 백테스팅
# ──────────────────────────────────────────────

def backtest_golden_cross(df: pd.DataFrame,
                           initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                           commission: float = DEFAULT_COMMISSION,
                           risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
                           annualization_days: int = 252) -> dict:
    """
    골든크로스(MA20 > MA60) 매수 / 데드크로스 매도 전략 백테스팅.

    Parameters
    ----------
    df                 : 기술적 지표가 추가된 DataFrame
    initial_capital    : 초기 자본금
    commission         : 수수료율 (기본 0.015%)
    risk_free_rate     : 무위험 수익률 (연, 소수점)
    annualization_days : 연환산 기준 거래일 수 (주식=252, 암호화폐=365)

    Note: 기간 말 미청산 포지션은 마지막 날 종가로 강제 청산 처리됨.
          이는 B&H 와의 공정한 비교를 위해 매도 수수료를 포함한다.
    """
    signals = df["MA_Signal"].values
    prices  = df["Close"].values
    dates   = df.index

    cash   = initial_capital
    shares = 0.0
    trades: list[dict] = []
    equity = np.empty(len(df))

    for i in range(len(df)):
        signal = signals[i]
        price  = float(prices[i])

        if signal == "golden" and cash > 0:
            fee    = cash * commission
            shares = (cash - fee) / price
            trades.append({
                "date": dates[i], "type": "buy",
                "price": price, "shares": shares, "fee": fee,
            })
            cash = 0.0

        elif signal == "dead" and shares > 0:
            sell_amount = shares * price
            fee         = sell_amount * commission
            cash        = sell_amount - fee
            trades.append({
                "date": dates[i], "type": "sell",
                "price": price, "shares": shares, "fee": fee,
            })
            shares = 0.0

        equity[i] = cash + shares * price

    # 기간 말 잔여 포지션 청산 (equity[-1] 은 실현가로 덮어씀)
    if shares > 0:
        last_price  = float(prices[-1])
        sell_amount = shares * last_price
        fee         = sell_amount * commission
        cash        = sell_amount - fee
        trades.append({
            "date": dates[-1], "type": "sell(final)",
            "price": last_price, "shares": shares, "fee": fee,
        })
        equity[-1] = cash

    equity_series = pd.Series(equity, index=dates)

    final_equity = float(equity_series.iloc[-1])
    total_ret    = (final_equity - initial_capital) / initial_capital

    n_days    = max(len(equity_series) - 1, 1)
    final_cum = final_equity / initial_capital if initial_capital > 0 else 1.0
    annual_ret = (final_cum ** (annualization_days / n_days)) - 1 if final_cum > 0 else 0.0

    daily_ret_bt = equity_series.pct_change().dropna()
    annual_vol   = daily_ret_bt.std() * math.sqrt(annualization_days)

    if annual_vol > 1e-8:
        sharpe = (annual_ret - risk_free_rate) / annual_vol
    else:
        sharpe = 0.0

    roll_max = equity_series.cummax()
    mdd      = ((equity_series - roll_max) / roll_max).min()

    buy_trades  = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if "sell" in t["type"]]
    # 수수료 차감 기준: 가격 차이 × 주수가 매수·매도 수수료 합산을 초과해야 실제 이익
    wins     = sum(1 for b, s in zip(buy_trades, sell_trades)
                   if (s["price"] - b["price"]) * b["shares"] > b["fee"] + s["fee"])
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
                           initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                           commission: float = DEFAULT_COMMISSION,
                           risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
                           annualization_days: int = 252) -> dict:
    """
    Buy & Hold 전략 성과 계산.

    [FIX 1차 B-4] MDD 기준 통일:
      기존: 종가 수익률 cumprod (수수료 미반영, 첫날 제외)
      수정: equity_series = shares * Close (수수료 반영 shares, 전 기간 커버)
      → 골든크로스 전략과 동일 기준으로 MDD 비교 가능
    """
    buy_price = float(df["Close"].iloc[0])
    buy_fee   = initial_capital * commission
    shares    = (initial_capital - buy_fee) / buy_price

    final_price  = float(df["Close"].iloc[-1])
    sell_amount  = shares * final_price
    sell_fee     = sell_amount * commission
    final_equity = sell_amount - sell_fee

    total_ret = (final_equity - initial_capital) / initial_capital

    n_days    = max(len(df) - 1, 1)
    annual_ret = (1 + total_ret) ** (annualization_days / n_days) - 1 \
                 if (1 + total_ret) > 0 else 0.0

    close_ret  = df["Close"].pct_change().dropna()
    annual_vol = close_ret.std() * math.sqrt(annualization_days)

    if annual_vol > 1e-8:
        sharpe = (annual_ret - risk_free_rate) / annual_vol
    else:
        sharpe = 0.0

    # [FIX 1차 B-4] equity_series 기반 MDD (골든크로스 방식 통일)
    equity_series = pd.Series(shares * df["Close"].values, index=df.index)
    roll_max      = equity_series.cummax()
    mdd           = ((equity_series - roll_max) / roll_max).min()

    return {
        "final_equity":    final_equity,
        "total_return":    total_ret,
        "annual_return":   annual_ret,
        "annual_vol":      annual_vol,
        "sharpe":          sharpe,
        "mdd":             mdd,
        "initial_capital": initial_capital,
    }


def print_backtest_report(bt_result: dict, bah_result: dict, ticker: str) -> None:
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
    print(f"  승률 (수수료 차감): {bt_result['win_rate']*100:>14.1f} %")

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
    summary = summarize_strategy_comparison(bt_result, bah_result)
    print(f"  수익률 차이       : {summary['return_diff']*100:>+14.2f} %")
    print(f"  샤프 비율 차이    : {summary['sharpe_diff']:>+15.3f}")
    print(f"  MDD 차이          : {summary['mdd_diff']*100:>+14.2f} %")
    print(f"\n  > 이 기간 우위 전략: {summary['winner']}")

    # 개별 거래 내역 (매수/매도 페어 + 손익)
    trades      = bt_result.get("trades", [])
    buy_trades  = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if "sell" in t["type"]]

    if buy_trades:
        print(f"\n[ 거래 내역 상세 ]")
        print(f"  {'#':>3}  {'매수일':^12}  {'매수가':>10}  {'매도일':^12}  {'매도가':>10}  {'손익(%)':>10}  {'결과':^4}")
        print(sep)
        for idx, (b, s) in enumerate(zip(buy_trades, sell_trades), start=1):
            buy_d  = (b["date"].strftime("%Y-%m-%d")
                      if hasattr(b["date"], "strftime") else str(b["date"])[:10])
            sell_d = (s["date"].strftime("%Y-%m-%d")
                      if hasattr(s["date"], "strftime") else str(s["date"])[:10])
            net_profit   = (s["price"] - b["price"]) * b["shares"] - b["fee"] - s["fee"]
            initial_cost = b["price"] * b["shares"] + b["fee"]
            net_pnl_pct  = net_profit / initial_cost * 100
            result       = "WIN " if net_profit > 0 else "LOSS"
            print(f"  {idx:>3}  {buy_d:^12}  {b['price']:>10,.1f}  "
                  f"{sell_d:^12}  {s['price']:>10,.1f}  {net_pnl_pct:>+9.2f}%  {result:^4}")
    print(f"{'='*60}\n")


def summarize_strategy_comparison(bt_result: dict, bah_result: dict) -> dict:
    """대시보드 표시용 백테스트 전략 비교 요약."""
    return_diff = bt_result["total_return"] - bah_result["total_return"]
    sharpe_diff = bt_result["sharpe"] - bah_result["sharpe"]
    mdd_diff    = bt_result["mdd"] - bah_result["mdd"]
    winner = ("골든크로스 전략" if bt_result["total_return"] > bah_result["total_return"]
              else "Buy & Hold 전략")
    return {
        "winner": winner,
        "return_diff": return_diff,
        "sharpe_diff": sharpe_diff,
        "mdd_diff": mdd_diff,
    }


def _print_analysis_reports(bt_result: dict, bah_result: dict, ticker: str) -> None:
    """Print final analysis reports after calculations and optional chart output."""
    print_backtest_report(bt_result, bah_result, ticker)


def _print_intermediate_analysis_reports(ticker: str,
                                         current_state: dict | None = None,
                                         stats: dict | None = None,
                                         benchmark_comparison: list[dict] | None = None) -> None:
    """Print intermediate console reports while preserving the existing order."""
    if current_state is not None:
        print_current_state(current_state, ticker)
    if stats is not None and benchmark_comparison is not None:
        print_benchmark_report(ticker, stats, benchmark_comparison)


def _print_annualization_basis(annualization_days: int) -> None:
    """Print the detected annualization basis."""
    print(f"[연환산 기준] {annualization_days}일 "
          f"({'암호화폐·24/7' if annualization_days == 365 else '주식·평일 거래'})")


def _print_indicator_calculation_done() -> None:
    """Print the technical indicator calculation completion message."""
    print("[지표 계산 완료] MA(5/20/60/120), RSI(14), MACD(12/26/9)")


def _make_cli_report_hooks() -> dict:
    """Return report hooks used by the CLI-style run_analysis flow."""
    return {
        "data_validation": _print_data_validation_reports,
        "annualization": _print_annualization_basis,
        "indicators": _print_indicator_calculation_done,
        "current_state": lambda ticker, summary: _print_intermediate_analysis_reports(
            ticker, current_state=summary
        ),
        "benchmark": lambda ticker, stats, rows: _print_intermediate_analysis_reports(
            ticker, stats=stats, benchmark_comparison=rows
        ),
    }


class _AnalysisResult(TypedDict):
    df:                   pd.DataFrame
    stats:                dict
    bt_result:            dict
    bah_result:           dict
    annualization_days:   int
    current_state:        dict
    benchmark_comparison: list
    strategy_summary:     dict
    data_quality:         dict
    external_price_check: dict


def _build_analysis_result(ticker: str,
                           period_years: float,
                           start: str | None,
                           end: str | None,
                           initial_capital: float,
                           commission: float,
                           risk_free_rate: float,
                           benchmarks: tuple[str, ...] | None,
                           benchmark_preset: str,
                           corr_window: int,
                           debug_benchmarks: bool,
                           debug_data_source: bool,
                           save_debug_columns: bool,
                           output_dir: str | Path,
                           report_hooks: dict | None = None) -> _AnalysisResult:
    """
    Build the analysis result dictionary from data collection and calculations.

    This function owns fetching data, computing metrics, indicators, benchmarks,
    and backtest results. User-facing side effects such as saving/showing the
    dashboard HTML and printing the final report stay in run_analysis().
    Optional report hooks are only used to preserve the existing CLI output order.
    """
    hooks = report_hooks or {}

    df = fetch_data(ticker, period_years=period_years, start=start, end=end,
                    debug_source=debug_data_source,
                    debug_columns_dir=output_dir if save_debug_columns else None)

    data_validation_hook = hooks.get("data_validation")
    if data_validation_hook is not None:
        data_quality, external_price_check = data_validation_hook(ticker, df)
    else:
        data_quality         = _compute_data_quality_stats(ticker, df)
        external_price_check = _compute_external_price_stats(ticker, df)

    ann_days = _detect_annualization_days(df)
    annualization_hook = hooks.get("annualization")
    if annualization_hook is not None:
        annualization_hook(ann_days)

    stats = compute_returns(df, risk_free_rate=risk_free_rate,
                            annualization_days=ann_days)

    df = add_indicators(df)
    indicators_hook = hooks.get("indicators")
    if indicators_hook is not None:
        indicators_hook()

    current_state = summarize_current_state(df)
    current_state_hook = hooks.get("current_state")
    if current_state_hook is not None:
        current_state_hook(ticker, current_state)

    selected_benchmarks = _select_benchmarks(ticker, benchmarks, benchmark_preset)
    benchmark_start = df.index[0].strftime("%Y-%m-%d")
    benchmark_end   = (df.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
    benchmark_comparison = compute_benchmark_comparison(
        stats,
        start=benchmark_start,
        end=benchmark_end,
        risk_free_rate=risk_free_rate,
        annualization_days=ann_days,
        benchmarks=selected_benchmarks,
        corr_window=corr_window,
        debug=debug_benchmarks,
        debug_source=debug_data_source,
        debug_columns_dir=output_dir if save_debug_columns else None,
    )
    benchmark_hook = hooks.get("benchmark")
    if benchmark_hook is not None:
        benchmark_hook(ticker, stats, benchmark_comparison)

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
    strategy_summary = summarize_strategy_comparison(bt_result, bah_result)

    return {
        "df":                   df,
        "stats":                stats,
        "bt_result":            bt_result,
        "bah_result":           bah_result,
        "annualization_days":   ann_days,
        "current_state":        current_state,
        "benchmark_comparison": benchmark_comparison,
        "strategy_summary":     strategy_summary,
        "data_quality":         data_quality,
        "external_price_check": external_price_check,
    }


# ──────────────────────────────────────────────
# 분석 파이프라인 통합
# ──────────────────────────────────────────────

def run_analysis(ticker: str,
                 period_years: float = DEFAULT_PERIOD_YEARS,
                 start: str = None,
                 end: str = None,
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 commission: float = DEFAULT_COMMISSION,
                 risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
                 show_chart: bool = True,
                 overwrite_html: bool = False,
                 output_dir: str | Path = ".",
                 benchmarks: tuple[str, ...] | None = None,
                 benchmark_preset: str = "auto",
                 show_excess_return: bool = True,
                 benchmark_display: str = "cumulative",
                 corr_window: int = 60,
                 show_marker_legend: bool = False,
                 show_signal_markers: bool = False,
                 show_benchmark_legend: bool = False,
                 debug_benchmarks: bool = False,
                 debug_data_source: bool = False,
                 save_debug_columns: bool = False) -> dict:
    """
    전체 분석 파이프라인 실행.

    Parameters
    ----------
    ticker           : 종목 코드 (예: AAPL, 005930, BTC-USD)
    period_years     : 분석 기간 (년, start/end 미지정 시 사용)
    start / end      : 직접 날짜 지정 'YYYY-MM-DD' (None 이면 period_years 사용)
    initial_capital  : 백테스팅 초기 자본금
    commission       : 매수·매도 수수료율 (기본 0.015%)
    risk_free_rate   : 무위험 수익률 (연, 소수점)
    show_chart       : Plotly 대시보드 표시 여부
    overwrite_html   : True → 기존 HTML 덮어쓰기 / False → 타임스탬프 파일명
    output_dir       : HTML 저장 디렉터리 (기본: 현재 디렉터리)
    benchmarks       : 명시적 벤치마크 튜플 (None 이면 preset 사용)
    benchmark_preset : 'auto'·'us'·'korea'·'crypto'·'off'
    show_excess_return: 초과수익선 표시 여부
    benchmark_display : 'all'·'cumulative'·'excess'
    corr_window      : 롤링 상관계수 창 (거래일)
    show_marker_legend: 시그널 마커 범례 표시 여부 (기본: False, CLI clean 모드와 동일)
    show_signal_markers: 시그널 마커 표시 여부 (기본: False, CLI clean 모드와 동일)
    show_benchmark_legend: 벤치마크 라인 범례 표시 여부
    debug_benchmarks : 벤치마크 첫/마지막 종가 검증 로그 표시 여부
    debug_data_source: yfinance 원본 컬럼 샘플 로그 표시 여부

    Returns
    -------
    dict {
        'df'                   : 지표가 추가된 DataFrame,
        'stats'                : compute_returns 결과 dict,
        'bt_result'            : 골든크로스 백테스팅 결과 dict,
        'bah_result'           : Buy & Hold 결과 dict,
        'annualization_days'   : 사용된 연환산 기준일,
        'current_state'        : 현재 기술적 상태 요약 dict,
        'benchmark_comparison' : 벤치마크 비교 결과 list[dict],
        'strategy_summary'     : 백테스트 전략 비교 요약 dict,
        'data_quality'         : 데이터 품질 검사 결과 dict,
        'external_price_check' : 외부 기준 가격 비교 결과 dict,
    }
    """
    # 0-a. 파라미터 검증
    _validate_params(ticker, initial_capital, commission, period_years)

    # 0-b. ticker 정규화 — 이하 모든 함수에 정규화된 ticker 전달
    # [FIX 1차 H-4] fetch_data 내부에서만 처리하던 것을 파이프라인 진입부로 이동
    ticker = ticker.strip()
    if not is_korean_ticker(ticker):
        ticker = ticker.upper()

    # 0-c. 의존성 확인 (정규화된 ticker 기준)
    # [FIX 1차 B-2] 원본 ticker 로 is_korean_ticker 호출하던 문제 해결
    _check_deps(need_plotly=show_chart,
                need_fdr=is_korean_ticker(ticker),
                need_yfinance=not is_korean_ticker(ticker))

    print(f"\n{'='*55}")
    print(f"  주식 분석 시스템 시작 | 티커: {ticker}")
    print(f"{'='*55}")

    result = _build_analysis_result(
        ticker=ticker,
        period_years=period_years,
        start=start,
        end=end,
        initial_capital=initial_capital,
        commission=commission,
        risk_free_rate=risk_free_rate,
        benchmarks=benchmarks,
        benchmark_preset=benchmark_preset,
        corr_window=corr_window,
        debug_benchmarks=debug_benchmarks,
        debug_data_source=debug_data_source,
        save_debug_columns=save_debug_columns,
        output_dir=output_dir,
        report_hooks=_make_cli_report_hooks(),
    )

    df                   = result["df"]
    stats                = result["stats"]
    bt_result            = result["bt_result"]
    bah_result           = result["bah_result"]
    current_state        = result["current_state"]
    benchmark_comparison = result["benchmark_comparison"]
    strategy_summary     = result["strategy_summary"]
    data_quality         = result["data_quality"]

    # 8. 시각화
    if show_chart:
        fig      = plot_dashboard(df, ticker, stats,
                                   current_state=current_state,
                                   benchmark_comparison=benchmark_comparison,
                                   show_excess_return=show_excess_return,
                                   benchmark_display=benchmark_display,
                                   show_marker_legend=show_marker_legend,
                                   show_signal_markers=show_signal_markers,
                                   strategy_summary=strategy_summary,
                                   data_quality=data_quality,
                                   show_benchmark_legend=show_benchmark_legend)
        _save_and_show_dashboard(fig, ticker, overwrite_html, output_dir)

    _print_analysis_reports(bt_result, bah_result, ticker)

    return result


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────

def main() -> None:
    """
    CLI 진입점.

    [FIX 1차 H-3] DEFAULT_* 상수를 함수 내 재정의 → 모듈 레벨 상수 참조

    사용 예시
    ---------
    python stock_analysis.py                              # 기본값(AAPL, 2년)
    python stock_analysis.py TSLA --years 3
    python stock_analysis.py 005930 --capital 5000000
    python stock_analysis.py BTC-USD --years 1 --no-chart
    python stock_analysis.py AAPL --start 2022-01-01 --end 2024-01-01
    python stock_analysis.py AAPL --overwrite
    python stock_analysis.py AAPL --output-dir ./reports
    """
    parser = argparse.ArgumentParser(
        prog="stock_analysis",
        description="주식·암호화폐 기술적 분석 및 골든크로스 백테스팅",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "ticker", nargs="?", default=DEFAULT_TICKER,
        help=f"종목 코드 (기본값: {DEFAULT_TICKER})\n예: AAPL  005930  BTC-USD  TSLA",
    )
    parser.add_argument(
        "--years", type=float, default=DEFAULT_PERIOD_YEARS, metavar="N",
        help=f"분석 기간 (년, 기본값: {DEFAULT_PERIOD_YEARS})",
    )
    parser.add_argument(
        "--start", type=str, default=None, metavar="YYYY-MM-DD",
        help="시작일 (지정 시 --years 무시)",
    )
    parser.add_argument(
        "--end", type=str, default=None, metavar="YYYY-MM-DD",
        help="종료일 (지정 시 --years 무시. 미지정 시 오늘)",
    )
    parser.add_argument(
        "--capital", type=float, default=DEFAULT_INITIAL_CAPITAL, metavar="N",
        help=f"초기 자본금 (기본값: {DEFAULT_INITIAL_CAPITAL:,})",
    )
    parser.add_argument(
        "--commission", type=float, default=DEFAULT_COMMISSION, metavar="F",
        help=f"수수료율 0~1 (기본값: {DEFAULT_COMMISSION})",
    )
    parser.add_argument(
        "--rfr", type=float, default=DEFAULT_RISK_FREE_RATE, metavar="F",
        help=f"무위험 수익률 (기본값: {DEFAULT_RISK_FREE_RATE})",
    )
    parser.add_argument(
        "--no-chart", action="store_true",
        help="차트 표시 및 HTML 저장 생략",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="기존 HTML 대시보드 덮어쓰기 (기본: 타임스탬프 파일명)",
    )
    # [FIX 2차 B-2] --output-dir 파라미터 추가
    parser.add_argument(
        "--output-dir", type=str, default=".", metavar="DIR",
        help="HTML 대시보드 저장 디렉터리 (기본값: 현재 디렉터리)",
    )
    parser.add_argument(
        "--benchmarks", type=_parse_benchmarks, default=None, metavar="A,B",
        help="쉼표 구분 벤치마크 티커 (기본: 자산 유형 자동 선택)",
    )
    parser.add_argument(
        "--benchmark-preset",
        choices=("auto", *BENCHMARK_PRESETS, "off"),
        default="auto",
        help="벤치마크 프리셋 (--benchmarks 미지정 시 사용)",
    )
    parser.add_argument(
        "--chart-mode",
        choices=("clean", "full"),
        default="clean",
        help="차트 표시 모드: clean(기본, 단순 표시) / full(마커·초과수익·범례 표시)",
    )
    parser.add_argument(
        "--no-excess-line", action="store_true",
        help="대시보드에서 초과수익선 숨김",
    )
    parser.add_argument(
        "--benchmark-display",
        choices=("all", "cumulative", "excess"),
        default=None,
        help="벤치마크 패널 표시 모드: all / cumulative / excess",
    )
    parser.add_argument(
        "--corr-window", type=_positive_int, default=60, metavar="N",
        help="롤링 상관계수 창 (거래일, 기본값: 60)",
    )
    parser.add_argument(
        "--hide-marker-legend", action="store_true",
        help="대시보드 범례에서 시그널 마커 항목 숨김 (기본값)",
    )
    parser.add_argument(
        "--show-marker-legend", action="store_true",
        help="대시보드 범례에 RSI/MACD/크로스 보조 마커 항목 표시",
    )
    parser.add_argument(
        "--hide-signal-markers", action="store_true",
        help="대시보드에서 RSI/MACD/크로스 보조 시그널 마커 숨김",
    )
    parser.add_argument(
        "--show-signal-markers", action="store_true",
        help="clean 모드에서도 RSI/MACD/크로스 보조 시그널 마커 표시",
    )
    parser.add_argument(
        "--show-benchmark-legend", action="store_true",
        help="대시보드 범례에 벤치마크 누적/초과수익 라인 항목 표시",
    )
    parser.add_argument(
        "--debug-benchmarks", action="store_true",
        help="벤치마크별 첫/마지막 종가와 행 수를 출력해 중복 데이터 여부 확인",
    )
    parser.add_argument(
        "--debug-data-source", action="store_true",
        help="벤치마크 yfinance 원본/정규화 컬럼 샘플 출력",
    )

    parser.add_argument(
        "--save-debug-columns", action="store_true",
        help="Save full raw yfinance column labels to --output-dir for debugging",
    )

    args = parser.parse_args()
    benchmark_display = args.benchmark_display
    if benchmark_display is None:
        benchmark_display = "all" if args.chart_mode == "full" else "cumulative"

    if args.hide_signal_markers:
        show_signal_markers = False
    elif args.show_signal_markers:
        show_signal_markers = True
    else:
        show_signal_markers = args.chart_mode == "full"

    show_marker_legend    = args.show_marker_legend and not args.hide_marker_legend
    show_benchmark_legend = args.show_benchmark_legend

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
            output_dir=args.output_dir,
            benchmarks=args.benchmarks,
            benchmark_preset=args.benchmark_preset,
            show_excess_return=not args.no_excess_line,
            benchmark_display=benchmark_display,
            corr_window=args.corr_window,
            show_marker_legend=show_marker_legend,
            show_signal_markers=show_signal_markers,
            show_benchmark_legend=show_benchmark_legend,
            debug_benchmarks=args.debug_benchmarks,
            debug_data_source=args.debug_data_source or args.save_debug_columns,
            save_debug_columns=args.save_debug_columns,
        )
    except (ValueError, ImportError) as e:
        print(f"\n[오류] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[예기치 못한 오류] {e}")
        raise


if __name__ == "__main__":
    main()
