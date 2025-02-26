import ccxt
import pandas as pd
from datetime import datetime, UTC
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, max_workers: int = 4, chunk_size: int = 200):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        
        self.binance = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'timeout': 30000,  # timeout 설정 추가
        })
        self.upbit = ccxt.upbit({
            'enableRateLimit': True,
            'timeout': 30000,  # timeout 설정 추가
        })
        
        Path('data/raw').mkdir(parents=True, exist_ok=True)

    def _get_formatted_time(self, timestamp_ms: int) -> str:
        return datetime.fromtimestamp(timestamp_ms/1000, tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')

    def _split_time_ranges(self, start_ts: int, end_ts: int) -> List[Tuple[int, int]]:
        chunk_ms = self.chunk_size * 3600 * 1000
        chunks = []
        current = start_ts
        while current < end_ts:
            next_chunk = min(current + chunk_ms, end_ts)
            chunks.append((current, next_chunk))
            current = next_chunk
        return chunks

    def _fetch_chunk(self, exchange, symbol: str, start_ts: int, end_ts: int, retry_count: int = 5) -> List:
        for attempt in range(retry_count):
            try:
                time.sleep(exchange.rateLimit * 2 / 1000)  # API 레이트 리밋 준수
                
                if exchange == self.upbit:
                    # Upbit의 경우 특별 처리
                    candles = exchange.fetch_ohlcv(
                        symbol, 
                        timeframe='1h',
                        since=start_ts,
                        limit=self.chunk_size
                    )
                else:
                    # Binance의 경우
                    candles = exchange.fetch_ohlcv(
                        symbol,
                        timeframe='1h',
                        since=start_ts,
                        limit=self.chunk_size
                    )

                if not candles:
                    logger.warning(f"No data returned for period starting at {self._get_formatted_time(start_ts)}")
                    if attempt < retry_count - 1:
                        time.sleep(2 ** attempt)  # exponential backoff
                        continue
                    return []

                # 시간 범위 내 데이터만 필터링
                filtered_candles = [c for c in candles if start_ts <= c[0] <= end_ts]
                return filtered_candles

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                    continue
                return []
        return []

    def fetch_parallel(self, exchange, symbol: str, chunks: List[Tuple[int, int]]) -> pd.DataFrame:
        all_candles = []
        retry_chunks = []
        
        with tqdm(total=len(chunks), desc=f"Fetching {symbol}") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._fetch_chunk, exchange, symbol, c[0], c[1]): c for c in chunks}
                
                for future in as_completed(futures):
                    chunk = futures[future]
                    try:
                        candles = future.result()
                        if candles:
                            all_candles.extend(candles)
                            start_date = self._get_formatted_time(candles[0][0])
                            end_date = self._get_formatted_time(candles[-1][0])
                            pbar.set_postfix({
                                'Period': f"{start_date[:16]} - {end_date[:16]}",
                                'Candles': len(candles)
                            })
                        else:
                            retry_chunks.append(chunk)
                    except Exception as e:
                        logger.error(f"Chunk error: {str(e)}")
                        retry_chunks.append(chunk)
                    finally:
                        pbar.update(1)

        # 실패한 청크 순차적 재시도
        if retry_chunks:
            logger.info(f"Retrying {len(retry_chunks)} failed chunks...")
            for chunk in retry_chunks:
                candles = self._fetch_chunk(exchange, symbol, chunk[0], chunk[1], retry_count=7)
                if candles:
                    all_candles.extend(candles)

        if not all_candles:
            raise ValueError(f"Failed to fetch any data for {symbol}")

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp').sort_index()
        
        # 중복 제거
        df = df[~df.index.duplicated(keep='first')]
        
        return df

    def fetch_and_save_data(self, start_ts: int, end_ts: int, save_path: str = 'data/raw/'):
        chunks = self._split_time_ranges(start_ts, end_ts)
        start_time = self._get_formatted_time(start_ts)
        end_time = self._get_formatted_time(end_ts)
        
        try:
            logger.info(f"Fetching data from {start_time} to {end_time}")
            
            logger.info("Fetching Binance perpetual futures data...")
            binance_df = self.fetch_parallel(self.binance, 'BTC/USDT:USDT', chunks)
            binance_df.to_csv(f'{save_path}binance_perpetual.csv')
            
            logger.info("Fetching Upbit spot data...")
            upbit_df = self.fetch_parallel(self.upbit, 'BTC/KRW', chunks)
            upbit_df.to_csv(f'{save_path}upbit_price.csv')
            
            # 데이터 정합성 검증
            logger.info(f"Binance records: {len(binance_df)}")
            logger.info(f"Upbit records: {len(upbit_df)}")
            
            # 공통 시간대 데이터만 유지
            common_times = binance_df.index.intersection(upbit_df.index)
            binance_df = binance_df.loc[common_times]
            upbit_df = upbit_df.loc[common_times]
            
            # 최종 저장
            binance_df.to_csv(f'{save_path}binance_perpetual.csv')
            upbit_df.to_csv(f'{save_path}upbit_price.csv')
            
            logger.info(f"Final aligned records: {len(binance_df)}")
            
            return binance_df, upbit_df
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            raise

def main():
    fetcher = DataFetcher(max_workers=4, chunk_size=200)
    
    try:
        start_ts = 1640995200 * 1000  # 2022-01-01 00:00:00 UTC
        end_ts = 1704067200 * 1000    # 2024-01-01 00:00:00 UTC
        
        binance_df, upbit_df = fetcher.fetch_and_save_data(start_ts, end_ts)
        logger.info("Data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")

if __name__ == "__main__":
    main()