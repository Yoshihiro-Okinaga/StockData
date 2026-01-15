from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd
import yfinance as yf


# =========================================
# 設定
# =========================================

@dataclass(frozen=True)
class SeriesConfig:
    name: str
    csv_path: Path
    ticker: str
    decimals: int                 # ← ファイルごとの小数桁
    interval: str = "1d"
    tz: str = "Asia/Tokyo"


DEFAULT_COLUMNS = ["日付", "曜日", "始値", "終値", "高値", "安値"]
PRICE_COLUMNS = ["始値", "終値", "高値", "安値"]


# =========================================
# ティッカー推測
# =========================================

_fx_pair_re = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_jp_stock_re = re.compile(r"^(?P<code>\d{4})_")  # 例: 1963_日揮ホールディングス

SPECIAL_TICKER_MAP = {
    "GOLD_USD": "GC=F",
    "XAU_USD": "GC=F",
    "OIL_USD": "CL=F",
    "US30_Futures": "YM=F",
    "JAPAN255_Futures": "NIY=F",
    "日経平均": "^N225",
}

def infer_ticker_from_stem(stem: str) -> Optional[str]:
    if stem in SPECIAL_TICKER_MAP:
        return SPECIAL_TICKER_MAP[stem]

    if _fx_pair_re.match(stem):
        base, quote = stem.split("_")
        return f"{base}{quote}=X"

    m = _jp_stock_re.match(stem)
    if m:
        return f"{m.group('code')}.T"

    return None


# =========================================
# 桁数（decimals）推定
# =========================================

def infer_decimals_from_existing_csv(csv_path: Path) -> Optional[int]:
    """
    既存CSVの文字列を見て、価格列の小数点以下最大桁を推定。
    例: 156.880 -> 3, 1.23456 -> 5, 1430 -> 0
    推定不能（価格列が全て空など）なら None
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, usecols=DEFAULT_COLUMNS)
    except Exception:
        return None

    max_dec = None
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        for v in s:
            v = v.strip()
            if not v:
                continue
            if "." in v:
                dec = len(v.split(".", 1)[1])
            else:
                dec = 0
            max_dec = dec if max_dec is None else max(max_dec, dec)

    return max_dec


def load_decimals_rules_from_stockfxlist(txt_path: Path) -> dict[str, int]:
    """
    StockFxList.txt の行末の数値（桁数）を読み取り、stem -> decimals を作る。
    例: "... EUR_USD.csv ... FX Investing 5" -> {"EUR_USD": 5}
    """
    rules: dict[str, int] = {}
    if not txt_path.exists():
        return rules

    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        parts = line.split()
        # 先頭がデータファイルパスのはず
        first = parts[0]
        stem = Path(first).stem  # 1963_日揮ホールディングス など

        # 行末に数字があれば桁数
        last = parts[-1]
        if last.isdigit():
            rules[stem] = int(last)

    return rules


def heuristic_default_decimals(stem: str) -> int:
    """
    最終手段（推定できないときのデフォルト）
    - 日本株/指数: 0
    - FX: JPY絡みは 3、それ以外は 5
    - Commodity/Index futures: 2（迷ったら2。既存CSV推定が基本）
    """
    if stem == "日経平均" or _jp_stock_re.match(stem):
        return 0

    if _fx_pair_re.match(stem):
        base, quote = stem.split("_")
        if base == "JPY" or quote == "JPY":
            return 3
        return 5

    # commodity / futures は銘柄ごとに違いがちなので控えめに2
    return 2


# =========================================
# CSV 読み書き
# =========================================

def load_price_csv(csv_path: Path, date_format: str = "%Y/%m/%d") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in DEFAULT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {missing} / path={csv_path}")

    df = df.copy()
    df["_date_dt"] = pd.to_datetime(df["日付"], format=date_format, errors="coerce")
    if df["_date_dt"].isna().all():
        raise ValueError(f"CSVの '日付' がパースできません: {csv_path}")
    return df


def save_price_csv(df: pd.DataFrame, csv_path: Path, decimals: int) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # decimals=0 なら %.0f（小数点なし）、NaNは空欄のまま
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format=f"%.{decimals}f")


def get_latest_date(df: pd.DataFrame) -> date:
    return df["_date_dt"].max().date()


# =========================================
# yfinance → 行生成（土日も行だけ作る）
# =========================================

def yfinance_history(ticker: str, start: date, end_exclusive: date, interval: str = "1d") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end_exclusive, interval=interval, auto_adjust=False)
    if hist is None:
        return pd.DataFrame()
    return hist


def _index_to_dates(hist_index: pd.DatetimeIndex, tz: str) -> list[date]:
    idx = pd.DatetimeIndex(hist_index)
    if idx.tz is not None:
        idx = idx.tz_convert(tz)
    return [ts.date() for ts in idx.to_pydatetime()]


def history_to_calendar_rows(
    hist: pd.DataFrame,
    start: date,
    up_to: date,
    tz: str,
    date_format: str,
    decimals: int,
) -> pd.DataFrame:
    cal = pd.date_range(pd.Timestamp(start), pd.Timestamp(up_to), freq="D")
    cal_dates = [d.date() for d in cal.to_pydatetime()]

    hist_map = {}
    if hist is not None and not hist.empty:
        hist_dates = _index_to_dates(hist.index, tz=tz)
        for d, (_, r) in zip(hist_dates, hist.iterrows()):
            hist_map[d] = {
                "Open": r.get("Open"),
                "Close": r.get("Close"),
                "High": r.get("High"),
                "Low": r.get("Low"),
            }

    rows = []
    for d in cal_dates:
        ts = pd.Timestamp(d)
        row = {
            "日付": ts.strftime(date_format),
            "曜日": ts.strftime("%a"),
            "始値": float("nan"),
            "終値": float("nan"),
            "高値": float("nan"),
            "安値": float("nan"),
        }
        if d in hist_map:
            row["始値"] = round(float(hist_map[d]["Open"]), decimals)
            row["終値"] = round(float(hist_map[d]["Close"]), decimals)
            row["高値"] = round(float(hist_map[d]["High"]), decimals)
            row["安値"] = round(float(hist_map[d]["Low"]), decimals)
        rows.append(row)

    return pd.DataFrame(rows, columns=DEFAULT_COLUMNS)


def merge_new_rows(existing_df: pd.DataFrame, new_rows: pd.DataFrame, date_format: str) -> tuple[pd.DataFrame, int]:
    base = existing_df.drop(columns=["_date_dt"], errors="ignore").copy()
    existing_dates = set(base["日付"].astype(str).tolist())

    add = new_rows[~new_rows["日付"].isin(existing_dates)].copy()
    added = len(add)

    out = pd.concat([add, base], ignore_index=True)
    out["_date_dt"] = pd.to_datetime(out["日付"], format=date_format, errors="coerce")
    out = out.sort_values("_date_dt", ascending=False).drop(columns=["_date_dt"])
    out = out[DEFAULT_COLUMNS]
    return out, added


# =========================================
# 1本埋める
# =========================================

def fill_csv_until_yesterday(
    config: SeriesConfig,
    today: Optional[date] = None,
    date_format: str = "%Y/%m/%d",
    quiet: bool = False,
    throttle_sec: float = 0.2,
) -> int:
    if today is None:
        today = date.today()
    up_to = today - timedelta(days=1)

    df = load_price_csv(config.csv_path, date_format=date_format)
    latest = get_latest_date(df)

    if latest >= up_to:
        if not quiet:
            print(f"[{config.name}] 追加不要: latest={latest} >= up_to={up_to}")
        out = df.drop(columns=["_date_dt"], errors="ignore")[DEFAULT_COLUMNS]
        save_price_csv(out, config.csv_path, config.decimals)
        return 0

    start = latest + timedelta(days=1)
    end_exclusive = up_to + timedelta(days=1)

    if throttle_sec > 0:
        time.sleep(throttle_sec)

    if not quiet:
        print(f"[{config.name}] {config.ticker} {start}〜{up_to} / decimals={config.decimals}")

    hist = yfinance_history(config.ticker, start=start, end_exclusive=end_exclusive, interval=config.interval)

    new_rows = history_to_calendar_rows(
        hist=hist,
        start=start,
        up_to=up_to,
        tz=config.tz,
        date_format=date_format,
        decimals=config.decimals,
    )

    out, added = merge_new_rows(df, new_rows, date_format=date_format)
    save_price_csv(out, config.csv_path, config.decimals)

    if not quiet:
        print(f"[{config.name}] 追加 {added} 行 / 保存: {config.csv_path}")
    return added


# =========================================
# フォルダ検索（FXCFDNightly / StockNightly のみ）
# =========================================

def build_configs_from_dirs(
    dirs: list[Path],
    stockfxlist_path: Optional[Path] = None,
) -> list[SeriesConfig]:
    rules = load_decimals_rules_from_stockfxlist(stockfxlist_path) if stockfxlist_path else {}

    configs: list[SeriesConfig] = []
    for d in dirs:
        for csv_path in sorted(d.rglob("*.csv")):
            stem = csv_path.stem
            ticker = infer_ticker_from_stem(stem)
            if ticker is None:
                continue

            # 1) 既存CSVから推定
            dec = infer_decimals_from_existing_csv(csv_path)
            # 2) 推定できないなら StockFxList の指定
            if dec is None and stem in rules:
                dec = rules[stem]
            # 3) それでも無いならヒューリスティック
            if dec is None:
                dec = heuristic_default_decimals(stem)

            configs.append(SeriesConfig(
                name=stem,
                csv_path=csv_path,
                ticker=ticker,
                decimals=int(dec),
            ))
    return configs


def fill_many(configs: Iterable[SeriesConfig], quiet: bool = False) -> None:
    for c in configs:
        try:
            fill_csv_until_yesterday(c, quiet=quiet)
        except Exception as e:
            print(f"[{c.name}] ERROR: {e}")


def main():
    base_dir = Path(r".")  # ← FXCFDNightly と StockNightly の親フォルダに合わせてください

    target_dirs = [
        base_dir / "FXCFDNightly",
        base_dir / "StockNightly",
    ]

    # StockFxList.txt を同階層に置くならこれでOK
    stockfxlist_path = base_dir / "StockFxList.txt"

    configs = build_configs_from_dirs(target_dirs, stockfxlist_path=stockfxlist_path)
    print(f"対象CSV: {len(configs)} 件")
    fill_many(configs)

if __name__ == "__main__":
    main()
