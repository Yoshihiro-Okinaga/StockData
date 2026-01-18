from __future__ import annotations

import json
import time
import os
import requests
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Iterable

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import pandas as pd
import yfinance as yf


# =========================
# 設定 & 定数
# =========================

BASE_COLUMNS = ["日付", "曜日", "始値", "終値", "高値", "安値"]
PRICE_COLUMNS = ["始値", "終値", "高値", "安値"]
STOCK_EXTRA_COLUMNS = ["出来高", "株式分割"]  # Stockのみ

# 12 Data API設定
TWELVEDATA_BASE_URL = "https://api.twelvedata.com/time_series"
# GitHub ActionsのSecrets、またはローカルの環境変数から読み込む
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")

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
# 設定読み込み (JSON)
# =========================

def resolve_local_csv_path(base_dir: Path, datafile_rel: str) -> Path:
    return base_dir / Path(datafile_rel)

def build_configs_from_json(base_dir: Path, json_path: Path) -> list[SeriesConfig]:
    """
    StockFxList.json を読み込み、SeriesConfigのリストを生成する
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSONファイルが見つかりません: {json_path}")

    # JSONロード
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    series_list = data.get("series", [])
    configs: list[SeriesConfig] = []

    for entry in series_list:
        # 必須項目の取得
        path_str = entry.get("path")
        s_type = entry.get("type")
        ticker = entry.get("ticker")
        
        if not (path_str and s_type and ticker):
            print(f"[SKIP] 必須項目不足: {entry}")
            continue

        csv_path = resolve_local_csv_path(base_dir, path_str)
        stem = csv_path.stem

        # フォーマット情報の取得
        fmt = entry.get("format", {})
        decimals = fmt.get("decimals") 
        
        configs.append(SeriesConfig(
            name=stem,
            csv_path=csv_path,
            ticker=ticker,
            stock_type=s_type,
            decimals=decimals,
        ))

    return configs


# =========================
# フォーマット関数
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
    if not csv_path.exists():
        # 新規作成用
        cols = columns_for_stock_type(stock_type)
        df = pd.DataFrame(columns=cols)
        df["_date_dt"] = pd.to_datetime([], errors="coerce")
        return df

    df = pd.read_csv(csv_path)

    need_cols = columns_for_stock_type(stock_type)
    # 足りない列は追加（互換性）
    for c in need_cols:
        if c not in df.columns:
            df[c] = float("nan")

    df = df.copy()
    df["_date_dt"] = pd.to_datetime(df["日付"], format=date_format, errors="coerce")
    
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
                out[col] = out[col].apply(lambda v: _fmt_fixed(v, decimals))

        # 出来高・株式分割の整形
        out["出来高"] = out["出来高"].apply(_fmt_int)
        out["株式分割"] = out["株式分割"].apply(_fmt_split)

        # ヘッダ作成
        header = ",".join(out_cols)
        lines = [header]
        
        for _, r in out.iterrows():
            split_val = str(r["株式分割"]).strip()
            if split_val == "":
                # 株式分割が無い → 最後の列を出さず、末尾カンマを消す
                fields = [str(r[c]) for c in out_cols[:-1]]
            else:
                # 株式分割がある → 全列出す
                fields = [str(r[c]) for c in out_cols]
            lines.append(",".join(fields))

        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        return

    # その他タイプの整形
    if decimals is None:
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
# データ取得ロジック (Yahoo / 12 Data)
# =========================

def _map_ticker_to_12data(yahoo_ticker: str) -> str:
    """
    Yahoo形式のティッカーを12 Data形式に変換するヘルパー
    例: USDJPY=X -> USD/JPY
    """
    # 余計なサフィックスを削除
    symbol = yahoo_ticker.replace("=X", "").replace("=F", "")

    # 通貨ペア (例: USDJPY -> USD/JPY)
    # 6文字かつ全てアルファベットの場合のみ分割を試みる
    if len(symbol) == 6 and symbol.isalpha():
        return f"{symbol[:3]}/{symbol[3:]}"

    return symbol


def fetch_yfinance(ticker: str, start: date, end_exclusive: date) -> pd.DataFrame:
    """Yahoo Financeから取得"""
    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end_exclusive, interval="1d", auto_adjust=False)
    return pd.DataFrame() if hist is None else hist


def fetch_12data(ticker: str, start: date, end_exclusive: date) -> pd.DataFrame:
    """12 Data APIから取得"""
    if not TWELVEDATA_API_KEY:
        print(f"[Warning] TWELVEDATA_API_KEY 未設定のため {ticker} をスキップします。")
        return pd.DataFrame()

    symbol = _map_ticker_to_12data(ticker)
    
    # 12 Dataは1リクエストで最大5000件
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end_exclusive.strftime("%Y-%m-%d"),
        "apikey": TWELVEDATA_API_KEY,
        "outputsize": 5000,
        "timezone": "Asia/Tokyo"
    }
    
    try:
        res = requests.get(TWELVEDATA_BASE_URL, params=params)
        data = res.json()
        
        if "status" in data and data["status"] == "error":
            print(f"[12 Data Error] {ticker} (as {symbol}): {data.get('message')}")
            return pd.DataFrame()
        
        if "values" not in data:
            return pd.DataFrame()
            
        # DataFrame化
        df = pd.DataFrame(data["values"])
        
        # カラム名のマッピング
        df = df.rename(columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close"
        })
        
        # 型変換
        df["Date"] = pd.to_datetime(df["Date"])
        cols = ["Open", "High", "Low", "Close"]
        for c in cols:
            df[c] = df[c].astype(float)
            
        df.set_index("Date", inplace=True)
        # 降順で来る場合があるので昇順に直す
        df.sort_index(inplace=True)
        
        # タイムゾーン処理
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Tokyo")
        else:
            df.index = df.index.tz_convert("Asia/Tokyo")
            
        return df
        
    except Exception as e:
        print(f"[12 Data Exception] {ticker}: {e}")
        return pd.DataFrame()


def get_history_unified(config: SeriesConfig, start: date, end_exclusive: date) -> pd.DataFrame:
    """タイプに応じてデータソースを振り分け"""
    
    # FXのみ 12 Data を使用 (10秒待機付き)
    if config.stock_type == "FX":
        time.sleep(10) 
        return fetch_12data(config.ticker, start, end_exclusive)
    
    # Stock, StockAverage, Commodity, Index は Yahoo を使用
    # (12 Dataの無料枠ではCommodity/Indexの制限が厳しいため)
    else:
        return fetch_yfinance(config.ticker, start, end_exclusive)


# =========================
# データ処理 & マージ
# =========================

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
    # 指定期間のカレンダー生成（土日含む）
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
        is_weekend = ts.weekday() >= 5  # 土曜(5) または 日曜(6)

        # デフォルトはNaN（空欄）
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

        # データが存在し、かつ「平日」である場合のみ値を入れる
        # (土日であればデータがあっても無視して空欄にする)
        if d in hist_map and not is_weekend:
            if stock_type == "Stock":
                o, c, h, l, v, s = hist_map[d]
                row["始値"] = float(o)
                row["終値"] = float(c)
                row["高値"] = float(h)
                row["安値"] = float(l)
                row["出来高"] = float(v) if v is not None else float("nan")
                
                try:
                    sv = float(s) if s is not None else float("nan")
                except Exception:
                    sv = float("nan")
                row["株式分割"] = float("nan") if (pd.isna(sv) or sv == 0.0) else sv
            else:
                o, c, h, l = hist_map[d]
                
                # 丸め処理
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
    # 既存にない日付だけ追加
    add = new_rows[~new_rows["日付"].isin(existing_dates)].copy()
    added = len(add)

    out = pd.concat([add, base], ignore_index=True)
    out["_date_dt"] = pd.to_datetime(out["日付"], format=date_format, errors="coerce")
    # 日付降順でソート
    out = out.sort_values("_date_dt", ascending=False).drop(columns=["_date_dt"])
    out = out[out_cols]
    return out, added


def fill_csv_until_yesterday(
    config: SeriesConfig,
    today: Optional[date] = None,
    date_format: str = "%Y/%m/%d",
    quiet: bool = False,
) -> int:
    if today is None:
        today = date.today()
    up_to = today - timedelta(days=1)

    # CSV読み込み
    try:
        df = load_price_csv(config.csv_path, stock_type=config.stock_type, date_format=date_format)
    except Exception as e:
        print(f"[{config.name}] CSV読込エラー: {e}")
        return 0

    # 最新日付の特定
    if df.empty or "_date_dt" not in df.columns or df["_date_dt"].dropna().empty:
         latest = date(2024, 1, 1)
    else:
        latest = df["_date_dt"].max().date()

    if latest >= up_to:
        if not quiet:
            print(f"[{config.name}] 更新不要 (最新: {latest})")
        return 0

    start = latest + timedelta(days=1)
    end_exclusive = up_to + timedelta(days=1)

    if not quiet:
        print(f"[{config.name}] 更新: {start} 〜 {up_to} (Source: {config.stock_type})")

    # データ取得
    hist = get_history_unified(config, start, end_exclusive)

    # 行生成（土日空欄）
    new_rows = history_to_calendar_rows(
        hist=hist,
        start=start,
        up_to=up_to,
        tz=config.tz,
        date_format=date_format,
        decimals=config.decimals,
        stock_type=config.stock_type,
    )

    # マージして保存
    out, added = merge_new_rows(df, new_rows, date_format=date_format, stock_type=config.stock_type)
    save_price_csv(out, config.csv_path, config.decimals, config.stock_type)

    if not quiet:
        print(f"[{config.name}] 追加 {added} 行 / 保存完了")
    return added


def fill_many(configs: Iterable[SeriesConfig], quiet: bool = False) -> None:
    for c in configs:
        try:
            fill_csv_until_yesterday(c, quiet=quiet)
        except Exception as e:
            print(f"[{c.name}] 処理中エラー: {e}")


def main():
    base_dir = Path(r".")
    stockfxlist_path = base_dir / "StockFxList.json"

    print("=== Stock Update Script Start ===")
    
    if TWELVEDATA_API_KEY:
        print("API Key: Detected")
    else:
        print("API Key: NOT Detected (FX/Commodity updates will skip)")

    try:
        configs = build_configs_from_json(base_dir, stockfxlist_path)
    except Exception as e:
        print(f"設定読み込みエラー: {e}")
        return

    print(f"更新対象: {len(configs)} 件")
    fill_many(configs)
    print("=== Finished ===")

if __name__ == "__main__":
    main()
