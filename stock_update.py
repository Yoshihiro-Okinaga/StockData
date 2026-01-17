from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Iterable

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import pandas as pd
import yfinance as yf


# =========================
# 設定
# =========================

BASE_COLUMNS = ["日付", "曜日", "始値", "終値", "高値", "安値"]
PRICE_COLUMNS = ["始値", "終値", "高値", "安値"]

STOCK_EXTRA_COLUMNS = ["出来高", "株式分割"]  # Stockのみ

@dataclass(frozen=True)
class SeriesConfig:
    name: str
    csv_path: Path
    ticker: str
    stock_type: str                 # Stock / StockAverage / FX / Commodity / Index ...
    decimals: Optional[int]         # 指定あり→固定0埋め、None→タイプ別の可変ルール
    interval: str = "1d"
    tz: str = "Asia/Tokyo"


# =========================
# ティッカー推測
# =========================

_fx_pair_re = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_jp_stock_re = re.compile(r"^(?P<code>\d{4})_")  # 例: 1963_日揮ホールディングス

SPECIAL_TICKER_MAP = {
    "日経平均": "^N225",
    "GOLD_USD": "GC=F",
    "OIL_USD": "CL=F",
    "JAPAN255_Futures": "NIY=F",
    "US30_Futures": "YM=F",
    "SILVER_USD": "SI=F",
    "PLATINUM_USD": "PL=F",
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


# =========================
# StockFxList.txt パース
# =========================

def parse_stockfxlist(stockfxlist_path: Path) -> list[tuple[str, str, Optional[int]]]:
    """
    戻り値: [(datafile_rel, stock_type, decimals_or_none), ...]
    例: ("FXCFDNightly/EUR_USD.csv", "FX", 5)
    """
    rows: list[tuple[str, str, Optional[int]]] = []
    text = stockfxlist_path.read_text(encoding="utf-8", errors="ignore")

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        datafile_rel = parts[0]
        stock_type = parts[1]

        decimals: Optional[int] = None
        # "FX Investing 5 ..." のような行に対応
        if len(parts) >= 4 and parts[3].isdigit():
            decimals = int(parts[3])

        rows.append((datafile_rel, stock_type, decimals))

    return rows


def default_decimals_from_list(stock_type: str, stem: str) -> int:
    """
    桁数指定が無いときのデフォルト（Stock/StockAverage はここでは使わない）
    """
    if stock_type == "FX":
        if "_JPY" in stem or stem.startswith("JPY_"):
            return 3
        return 5
    return 2  # Commodity / Index など


def resolve_local_csv_path(base_dir: Path, datafile_rel: str) -> Path:
    return base_dir / Path(datafile_rel)


def build_configs_from_stockfxlist(base_dir: Path, stockfxlist_path: Path) -> list[SeriesConfig]:
    entries = parse_stockfxlist(stockfxlist_path)
    configs: list[SeriesConfig] = []

    for datafile_rel, stock_type, decimals_opt in entries:
        csv_path = resolve_local_csv_path(base_dir, datafile_rel)
        stem = csv_path.stem

        ticker = infer_ticker_from_stem(stem)
        if ticker is None:
            print(f"[SKIP] ticker推測不可: {stem} ({datafile_rel})")
            continue

        # ここが今回の肝：
        # - リストに桁数指定があるものは固定（0埋め）
        # - 日経平均(StockAverage) は 3桁固定 0埋め
        # - 個別株(Stock) は “整数→整数 / 小数→最大1桁で四捨五入” の可変
        # - それ以外で桁数指定なしは従来デフォルト
        if decimals_opt is not None:
            decimals: Optional[int] = int(decimals_opt)
        elif stock_type == "StockAverage":
            decimals = None
        elif stock_type == "Stock":
            decimals = None
        else:
            decimals = int(default_decimals_from_list(stock_type, stem))

        configs.append(SeriesConfig(
            name=stem,
            csv_path=csv_path,
            ticker=ticker,
            stock_type=stock_type,
            decimals=decimals,
        ))

    return configs


# =========================
# フォーマット
# =========================

def _fmt_variable_number(x) -> str:
    """末尾0を落とす（一般用途）"""
    if pd.isna(x):
        return ""
    s = str(x)
    try:
        d = Decimal(s)
    except InvalidOperation:
        return s
    s2 = format(d, "f")
    if "." in s2:
        s2 = s2.rstrip("0").rstrip(".")
    return s2


def _fmt_stock_price_0_or_1dp(x) -> str:
    """
    Stock用：
    - 整数なら整数
    - 小数なら最大1桁、四捨五入
    """
    if pd.isna(x):
        return ""
    s = str(x)
    try:
        d = Decimal(s)
    except InvalidOperation:
        return s

    # 整数判定
    if d == d.to_integral_value():
        return str(int(d))

    # 1桁で四捨五入
    q = d.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    s2 = format(q, "f")
    if "." in s2:
        s2 = s2.rstrip("0").rstrip(".")
    return s2


def _fmt_int(x) -> str:
    """出来高用。NaNは空欄。"""
    if pd.isna(x):
        return ""
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def _fmt_split(x) -> str:
    """株式分割用：0/NaNは空欄、それ以外は自然表示"""
    if pd.isna(x):
        return ""
    try:
        v = float(x)
        if v == 0.0:
            return ""
    except Exception:
        s = str(x).strip()
        if s in ("", "0", "0.0", "0.00", "0.000"):
            return ""
        return s
    return _fmt_variable_number(x)


def _fmt_fixed(x, decimals: int) -> str:
    """固定桁（0埋め）用。NaNは空欄。"""
    if pd.isna(x):
        return ""
    return f"{float(x):.{decimals}f}"


def _fmt_round_3_and_pad(x) -> str:
    if pd.isna(x):
        return ""
    d = Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{d:.3f}"

# =========================
# CSV 読み書き
# =========================

def columns_for_stock_type(stock_type: str) -> list[str]:
    if stock_type == "Stock":
        return BASE_COLUMNS + STOCK_EXTRA_COLUMNS
    return BASE_COLUMNS


def load_price_csv(csv_path: Path, stock_type: str, date_format: str = "%Y/%m/%d") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    need_cols = columns_for_stock_type(stock_type)
    # 足りない列は追加（互換性）
    for c in need_cols:
        if c not in df.columns:
            df[c] = float("nan")

    df = df.copy()
    df["_date_dt"] = pd.to_datetime(df["日付"], format=date_format, errors="coerce")
    if df["_date_dt"].isna().all():
        raise ValueError(f"CSVの '日付' がパースできません: {csv_path}")

    return df


def save_price_csv(df: pd.DataFrame, csv_path: Path, decimals: Optional[int], stock_type: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    out_cols = columns_for_stock_type(stock_type)
    out = df[out_cols].copy()

    # Stock追加列の整形
    if stock_type == "Stock":
        # 価格列の整形（可変 or 固定）
        for col in PRICE_COLUMNS:
            if decimals is None:
                out[col] = out[col].apply(_fmt_stock_price_0_or_1dp)
            else:
                out[col] = out[col].apply(lambda v: _fmt_fixed(v, decimals))  # 0埋め固定

        # 出来高・株式分割の整形
        out["出来高"] = out["出来高"].apply(_fmt_int)
        out["株式分割"] = out["株式分割"].apply(_fmt_split)  # 0/NaN -> ""

        # ヘッダは必ず 8列（株式分割の文字は必ず入る）
        header = ",".join(out_cols)

        lines = [header]
        for _, r in out.iterrows():
            split_val = str(r["株式分割"]).strip()

            if split_val == "":
                # ★株式分割が無い → 最後の列を出さず、末尾カンマを消す
                fields = [str(r[c]) for c in out_cols[:-1]]
            else:
                # ★株式分割がある → 8列全部出す
                fields = [str(r[c]) for c in out_cols]

            lines.append(",".join(fields))

        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        return

    # 価格整形
    if decimals is None:
        # Stock: 0 or 1dp ルール
        if stock_type == "StockAverage":
            for col in PRICE_COLUMNS:
                out[col] = out[col].apply(_fmt_round_3_and_pad)
        else:
            for col in PRICE_COLUMNS:
                out[col] = out[col].apply(_fmt_variable_number)

        out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        # 固定桁（0埋め）
        out.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format=f"%.{decimals}f")


# =========================
# yfinance → 行生成（土日も行を作る）
# =========================

def yfinance_history(ticker: str, start: date, end_exclusive: date, interval: str = "1d") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end_exclusive, interval=interval, auto_adjust=False)
    return pd.DataFrame() if hist is None else hist


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
    decimals: Optional[int],
    stock_type: str,
) -> pd.DataFrame:
    cal = pd.date_range(pd.Timestamp(start), pd.Timestamp(up_to), freq="D")
    cal_dates = [d.date() for d in cal.to_pydatetime()]

    hist_map = {}
    if not hist.empty:
        hist_dates = _index_to_dates(hist.index, tz=tz)
        for d, (_, r) in zip(hist_dates, hist.iterrows()):
            if stock_type == "Stock":
                hist_map[d] = (
                    r.get("Open"), r.get("Close"), r.get("High"), r.get("Low"),
                    r.get("Volume"), r.get("Stock Splits"),
                )
            else:
                hist_map[d] = (r.get("Open"), r.get("Close"), r.get("High"), r.get("Low"))

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
        if stock_type == "Stock":
            row["出来高"] = float("nan")
            row["株式分割"] = float("nan")

        if d in hist_map:
            if stock_type == "Stock":
                o, c, h, l, v, s = hist_map[d]
                row["始値"] = float(o)
                row["終値"] = float(c)
                row["高値"] = float(h)
                row["安値"] = float(l)
                row["出来高"] = float(v) if v is not None else float("nan")
                # 分割は0なら空欄にする（保存側でも空欄化）
                try:
                    sv = float(s) if s is not None else float("nan")
                except Exception:
                    sv = float("nan")
                row["株式分割"] = float("nan") if (pd.isna(sv) or sv == 0.0) else sv
            else:
                o, c, h, l = hist_map[d]
                # 固定桁のもの（例: 日経平均3桁）はここで丸めておく
                if decimals is None:
                    row["始値"] = float(o)
                    row["終値"] = float(c)
                    row["高値"] = float(h)
                    row["安値"] = float(l)
                else:
                    row["始値"] = round(float(o), decimals)
                    row["終値"] = round(float(c), decimals)
                    row["高値"] = round(float(h), decimals)
                    row["安値"] = round(float(l), decimals)

        rows.append(row)

    return pd.DataFrame(rows, columns=columns_for_stock_type(stock_type))


def merge_new_rows(existing_df: pd.DataFrame, new_rows: pd.DataFrame, date_format: str, stock_type: str) -> tuple[pd.DataFrame, int]:
    out_cols = columns_for_stock_type(stock_type)

    base = existing_df.drop(columns=["_date_dt"], errors="ignore").copy()
    base = base[out_cols].copy()

    existing_dates = set(base["日付"].astype(str).tolist())
    add = new_rows[~new_rows["日付"].isin(existing_dates)].copy()
    added = len(add)

    out = pd.concat([add, base], ignore_index=True)
    out["_date_dt"] = pd.to_datetime(out["日付"], format=date_format, errors="coerce")
    out = out.sort_values("_date_dt", ascending=False).drop(columns=["_date_dt"])
    out = out[out_cols]
    return out, added


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

    df = load_price_csv(config.csv_path, stock_type=config.stock_type, date_format=date_format)
    latest = df["_date_dt"].max().date()

    if latest >= up_to:
        if not quiet:
            print(f"[{config.name}] 追加不要")
        out = df.drop(columns=["_date_dt"], errors="ignore")
        save_price_csv(out, config.csv_path, config.decimals, config.stock_type)
        return 0

    start = latest + timedelta(days=1)
    end_exclusive = up_to + timedelta(days=1)

    if throttle_sec > 0:
        time.sleep(throttle_sec)

    if not quiet:
        dec_str = "variable" if config.decimals is None else str(config.decimals)
        print(f"[{config.name}] {config.ticker} {start}〜{up_to} / type={config.stock_type} / decimals={dec_str}")

    hist = yfinance_history(config.ticker, start=start, end_exclusive=end_exclusive, interval=config.interval)
    new_rows = history_to_calendar_rows(
        hist=hist,
        start=start,
        up_to=up_to,
        tz=config.tz,
        date_format=date_format,
        decimals=config.decimals,
        stock_type=config.stock_type,
    )

    out, added = merge_new_rows(df, new_rows, date_format=date_format, stock_type=config.stock_type)
    save_price_csv(out, config.csv_path, config.decimals, config.stock_type)

    if not quiet:
        print(f"[{config.name}] 追加 {added} 行 / 保存: {config.csv_path}")
    return added


def fill_many(configs: Iterable[SeriesConfig], quiet: bool = False) -> None:
    for c in configs:
        try:
            fill_csv_until_yesterday(c, quiet=quiet)
        except Exception as e:
            print(f"[{c.name}] ERROR: {e}")


def main():
    # StockFxList.txt に書いてある相対パスの「親フォルダ」を指定
    base_dir = Path(r".")  # 例: Path(r"D:\Okinaga\YourRoot")
    stockfxlist_path = base_dir / "StockFxList.txt"

    configs = build_configs_from_stockfxlist(base_dir, stockfxlist_path)
    print(f"更新対象（StockFxList.txt 限定）: {len(configs)} 件")
    fill_many(configs)

if __name__ == "__main__":
    main()
