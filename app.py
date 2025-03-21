from flask import Flask, render_template, request, session, jsonify
import requests
import random
import json
import logging
import os
import pickle
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

logging.basicConfig(level=logging.DEBUG)

# Ensure cache directory exists
os.makedirs("cache", exist_ok=True)

def fetch_chart_data(coin=None, timeframe=None, limit=8760):  # Set a high limit to accommodate 1h timeframe (365 * 24 = 8760)
    # Define available coins and timeframes
    all_coins = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cosmos', 'ripple', 'litecoin', 'chainlink']
    hourly_coins = ['bitcoin', 'ethereum', 'binancecoin', 'solana']
    timeframes = {'1h': 2, '4h': 14, '1d': 60, '1w': 180}  # Kept for compatibility, but we'll fetch 365 days
    
    # Select timeframe if not provided
    timeframe = timeframe or random.choice(list(timeframes.keys()))
    logging.debug(f"Selected timeframe: {timeframe}")
    
    # Select coin based on timeframe restrictions
    if coin is None:
        if timeframe == '1h':
            coin = random.choice(hourly_coins)
            logging.debug(f"Selected coin for 1h timeframe: {coin}")
        else:
            coin = random.choice(all_coins)
            logging.debug(f"Selected coin for non-1h timeframe: {coin}")
    
    # Use caching to avoid repeated API calls
    cache_file = f"cache/{coin}_ohlc_365days.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            raw_data = pickle.load(f)
        print(f"Loaded {coin} data from cache (CoinGecko OHLC)")
        logging.debug(f"Loaded {coin} data from cache")
    else:
        days = 365  # Fetch 365 days of data
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency=usd&days={days}"
        headers = {"x-cg-demo-api-key": "CG-X9rKSiVeFyMS6FPbUCaFw4Lc"}
        print(f"Fetching data for {coin} ({timeframe}) from {url} with days={days}")
        logging.info(f"Selected coin: {coin} for timeframe: {timeframe}")
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            print(f"API Response Status: {response.status_code}")
            logging.debug(f"API response status code: {response.status_code}")
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                logging.error(f"API Error: {response.status_code} - {response.text}")
                return [], coin, timeframe
            
            raw_data = response.json()
            if not raw_data:
                print("No OHLC data received from CoinGecko")
                logging.error("No OHLC data received from CoinGecko")
                return [], coin, timeframe
            
            # Cache the raw 1-day OHLC data
            with open(cache_file, 'wb') as f:
                pickle.dump(raw_data, f)
            print(f"Saved {coin} data to cache (CoinGecko OHLC)")
            logging.debug(f"Saved {coin} data to cache")
        
        except requests.exceptions.RequestException as e:
            print(f"Exception fetching data: {e}")
            logging.error(f"Exception in fetch_chart_data: {e}")
            return [], coin, timeframe
    
    print(f"Received {len(raw_data)} daily candles from CoinGecko")
    logging.debug(f"Received {len(raw_data)} daily candles from CoinGecko")
    logging.debug(f"Sample of raw data: {raw_data[-5:]}")
    
    # Convert 1-day candles into the desired timeframe
    chart_data = []
    if timeframe == '1h':
        # Convert each 1-day candle into 24 hourly candles
        for row in raw_data:
            base_time = int(row[0] / 1000)  # Convert milliseconds to seconds
            open_price = float(row[1])
            close_price = float(row[4])
            high_price = float(row[2])
            low_price = float(row[3])
            
            # Linearly interpolate prices over 24 hours
            for hour in range(24):
                time = base_time + hour * 3600
                # Interpolate the price for this hour
                price_ratio = hour / 24
                interpolated_price = open_price + (close_price - open_price) * price_ratio
                # Scale high and low proportionally
                high = high_price * (1 + (hour / 24) * (close_price / open_price - 1)) if open_price != 0 else high_price
                low = low_price * (1 + (hour / 24) * (close_price / open_price - 1)) if open_price != 0 else low_price
                candle = {
                    'time': time,
                    'open': interpolated_price if hour == 0 else chart_data[-1]['close'],
                    'high': high,
                    'low': low,
                    'close': interpolated_price
                }
                chart_data.append(candle)
        logging.debug(f"Converted to 1h timeframe: {len(chart_data)} hourly candles")
    elif timeframe == '4h':
        # Convert each 1-day candle into 6 4-hourly candles
        for row in raw_data:
            base_time = int(row[0] / 1000)
            open_price = float(row[1])
            close_price = float(row[4])
            high_price = float(row[2])
            low_price = float(row[3])
            
            for segment in range(6):
                time = base_time + segment * 14400  # 4 hours = 14400 seconds
                price_ratio = segment / 6
                interpolated_price = open_price + (close_price - open_price) * price_ratio
                high = high_price * (1 + (segment / 6) * (close_price / open_price - 1)) if open_price != 0 else high_price
                low = low_price * (1 + (segment / 6) * (close_price / open_price - 1)) if open_price != 0 else low_price
                candle = {
                    'time': time,
                    'open': interpolated_price if segment == 0 else chart_data[-1]['close'],
                    'high': high,
                    'low': low,
                    'close': interpolated_price
                }
                chart_data.append(candle)
        logging.debug(f"Converted to 4h timeframe: {len(chart_data)} 4-hourly candles")
    elif timeframe == '1d':
        # Use the 1-day candles as-is
        for row in raw_data:
            candle = {
                'time': int(row[0] / 1000),
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4])
            }
            chart_data.append(candle)
        logging.debug(f"Using 1d timeframe: {len(chart_data)} daily candles")
    elif timeframe == '1w':
        # Aggregate 7 days into 1 weekly candle
        weekly_candles = {}
        for row in raw_data:
            timestamp = int(row[0] / 1000)
            week_start = timestamp - (timestamp % (7 * 86400))
            if week_start not in weekly_candles:
                weekly_candles[week_start] = {
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4])
                }
            else:
                candle = weekly_candles[week_start]
                candle['high'] = max(candle['high'], float(row[2]))
                candle['low'] = min(candle['low'], float(row[3]))
                candle['close'] = float(row[4])
        
        for week_start, candle in sorted(weekly_candles.items()):
            chart_data.append({
                'time': week_start,
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close']
            })
        logging.debug(f"Converted to 1w timeframe: {len(chart_data)} weekly candles")
    
    # Apply the limit
    candles_to_take = min(limit, len(chart_data))
    chart_data = chart_data[-candles_to_take:]
    print(f"Limit: {limit}, Data length: {len(chart_data)}, Candles to take: {candles_to_take}")
    print(f"Fetched {len(chart_data)} candles for {coin} ({timeframe})")
    print(f"Sample of processed chart data: {chart_data[-5:]}")
    logging.debug(f"Final chart data length: {len(chart_data)}")
    logging.debug(f"Sample of processed chart data: {chart_data[-5:]}")
    
    return chart_data, coin, timeframe

def detect_swing_points(data, lookback=5, timeframe='4h', significance_threshold=0.01):
    lookback_map = {'1h': 8, '4h': 5, '1d': 3, '1w': 2}
    lookback = lookback_map.get(timeframe, 5)

    swing_points = {'highs': [], 'lows': []}
    price_range = max(c['high'] for c in data) - min(c['low'] for c in data) if data else 1
    min_price_diff = price_range * significance_threshold

    for i in range(lookback, len(data) - lookback):
        current = data[i]
        before = [c['high'] for c in data[i - lookback:i]]
        after = [c['high'] for c in data[i + 1:i + 1 + lookback]]
        if current['high'] > max(before) and current['high'] > max(after):
            window_lows = [c['low'] for c in data[i - lookback:i + 1 + lookback]]
            lowest_low = min(window_lows)
            price_diff = current['high'] - lowest_low
            if price_diff >= min_price_diff:
                swing_points['highs'].append({'time': current['time'], 'price': current['high']})

        before_lows = [c['low'] for c in data[i - lookback:i]]
        after_lows = [c['low'] for c in data[i + 1:i + 1 + lookback]]
        if current['low'] < min(before_lows) and current['low'] < min(after_lows):
            window_highs = [c['high'] for c in data[i - lookback:i + 1 + lookback]]
            highest_high = max(window_highs)
            price_diff = highest_high - current['low']
            if price_diff >= min_price_diff:
                swing_points['lows'].append({'time': current['time'], 'price': current['low']})

    print(f"Detected {len(swing_points['highs'])} swing highs and {len(swing_points['lows'])} swing lows")
    return swing_points

def determine_trend(data):
    if len(data) < 10:
        return 'sideways'
    
    recent_data = data[-10:]
    highs = [c['high'] for c in recent_data]
    lows = [c['low'] for c in recent_data]
    
    is_uptrend = all(highs[i] < highs[i + 1] for i in range(len(highs) - 1)) and \
                 all(lows[i] < lows[i + 1] for i in range(len(lows) - 1))
    
    is_downtrend = all(highs[i] > highs[i + 1] for i in range(len(highs) - 1)) and \
                   all(lows[i] > lows[i + 1] for i in range(len(lows) - 1))
    
    if is_uptrend:
        return 'uptrend'
    elif is_downtrend:
        return 'downtrend'
    else:
        return 'sideways'

def detect_fair_value_gaps(data, gap_type='bullish'):
    """
    Detect Fair Value Gaps (FVGs) based on precise visual pattern.
    
    For a Bullish FVG:
    - First candle must be bearish (close < open)
    - Third candle must be bullish (close > open)
    - NO OVERLAP between first candle's high and third candle's low
      (i.e., third candle's low > first candle's high)
    
    For a Bearish FVG:
    - First candle must be bullish (close > open)
    - Third candle must be bearish (close < open)
    - NO OVERLAP between first candle's low and third candle's high
      (i.e., third candle's high < first candle's low)
    
    Args:
        data: List of candle data (OHLC)
        gap_type: 'bullish' or 'bearish'
        
    Returns:
        List of detected FVGs
    """
    if not data or len(data) < 3:
        return []
    
    gaps = []
    logging.debug(f"Analyzing {len(data)} candles for {gap_type} FVGs")
    
    # Process each set of three consecutive candles
    for i in range(len(data) - 2):
        candle1 = data[i]
        candle2 = data[i+1]  # Middle candle
        candle3 = data[i+2]
        
        # Check if candles are bullish or bearish
        candle1_is_bearish = candle1['close'] < candle1['open']
        candle1_is_bullish = candle1['close'] > candle1['open']
        candle3_is_bearish = candle3['close'] < candle3['open']
        candle3_is_bullish = candle3['close'] > candle3['open']
        
        # Check for exact pattern match based on the visuals provided
        if gap_type == 'bullish':
            # 1. First candle must be bearish
            if not candle1_is_bearish:
                continue
                
            # 2. Third candle must be bullish
            if not candle3_is_bullish:
                continue
                
            # 3. NO OVERLAP - Third candle's low must be above first candle's high
            if not (candle3['low'] > candle1['high']):
                continue
                
            # We have a valid bullish FVG
            gap = {
                'startTime': candle1['time'],
                'endTime': candle3['time'],
                'topPrice': candle3['low'],  # Top of the gap
                'bottomPrice': candle1['high'],  # Bottom of the gap
                'type': 'bullish',
                'size': candle3['low'] - candle1['high']
            }
            
            logging.debug(f"Found bullish FVG: {gap}")
            gaps.append(gap)
            
        elif gap_type == 'bearish':
            # 1. First candle must be bullish
            if not candle1_is_bullish:
                continue
                
            # 2. Third candle must be bearish
            if not candle3_is_bearish:
                continue
                
            # 3. NO OVERLAP - Third candle's high must be below first candle's low
            if not (candle3['high'] < candle1['low']):
                continue
                
            # We have a valid bearish FVG
            gap = {
                'startTime': candle1['time'],
                'endTime': candle3['time'],
                'topPrice': candle1['low'],  # Top of the gap
                'bottomPrice': candle3['high'],  # Bottom of the gap
                'type': 'bearish',
                'size': candle1['low'] - candle3['high']
            }
            
            logging.debug(f"Found bearish FVG: {gap}")
            gaps.append(gap)
    
    # Sort gaps by size (largest first)
    gaps.sort(key=lambda x: x['size'], reverse=True)
    
    logging.debug(f"Detected {len(gaps)} {gap_type} FVGs")
    return gaps
    
@app.route('/charting_exams')
def charting_exams():
    session.pop('exam_data', None)
    return render_template('index.html')

@app.route('/charting_exam/swing_analysis', methods=['GET'])
def swing_analysis():
    if 'exam_data' not in session:
        session['exam_data'] = {
            'chart_count': 1,
            'scores': [],
            'chart_data': None,
            'coin': None,
            'timeframe': None
        }

    exam_data = session['exam_data']

    chart_data, coin, timeframe = fetch_chart_data()
    if not chart_data:
        return render_template(
            'swing_analysis.html',
            chart_data=[],
            progress={'chart_count': exam_data['chart_count']},
            symbol="ERROR",
            timeframe=timeframe,
            error="Failed to fetch chart data."
        )
    
    exam_data['chart_data'] = chart_data
    exam_data['coin'] = coin
    exam_data['timeframe'] = timeframe
    session['exam_data'] = exam_data

    return render_template(
        'swing_analysis.html',
        chart_data=chart_data,
        progress={'chart_count': exam_data['chart_count']},
        symbol=coin.upper(),
        timeframe=timeframe
    )

@app.route('/charting_exam/fibonacci_retracement', methods=['GET', 'POST'])
def fibonacci_retracement():
    if request.args.get('reset') == 'true':
        session.pop('exam_data', None)

    if 'exam_data' not in session:
        session['exam_data'] = {
            'chart_count': 1,
            'fibonacci_part': 1,
            'scores': [],
            'chart_data': None,
            'coin': None,
            'timeframe': None
        }

    exam_data = session['exam_data']

    if request.method == 'GET':
        chart_data, coin, timeframe = fetch_chart_data()
        if not chart_data:
            return render_template(
                'fibonacci_retracement.html',
                chart_data=[],
                progress={'chart_count': exam_data['chart_count'], 'fibonacci_part': exam_data['fibonacci_part']},
                symbol="ERROR",
                timeframe=timeframe,
                error="Failed to fetch chart data."
            )
        
        exam_data['chart_data'] = chart_data
        exam_data['coin'] = coin
        exam_data['timeframe'] = timeframe
        exam_data['fibonacci_part'] = 1
        session['exam_data'] = exam_data

        logging.debug(f"Fibonacci Retracement - Stored Chart Data Length: {len(chart_data)}")
        logging.debug(f"Fibonacci Retracement - Stored Chart Data Sample: {chart_data[:5]}")

        return render_template(
            'fibonacci_retracement.html',
            chart_data=chart_data,
            progress={'chart_count': exam_data['chart_count'], 'fibonacci_part': exam_data['fibonacci_part']},
            symbol=coin.upper(),
            timeframe=timeframe
        )
    return jsonify({'message': 'Fibonacci Retracement POST received'})

@app.route('/charting_exam/fair_value_gaps', methods=['GET', 'POST'])
def fair_value_gaps():
    if 'exam_data' not in session:
        session['exam_data'] = {
            'chart_count': 1,
            'fvg_part': 1,
            'scores': [],
            'chart_data': None,
            'coin': None,
            'timeframe': None
        }

    exam_data = session['exam_data']

    if request.method == 'GET':
        chart_data, coin, timeframe = fetch_chart_data()
        if not chart_data:
            return render_template(
                'fair_value_gaps.html',
                chart_data=[],
                progress={'chart_count': exam_data['chart_count'], 'fvg_part': exam_data['fvg_part']},
                symbol="ERROR",
                timeframe=timeframe,
                error="Failed to fetch chart data."
            )
        
        exam_data['chart_data'] = chart_data
        exam_data['coin'] = coin
        exam_data['timeframe'] = timeframe
        exam_data['fvg_part'] = 1
        session['exam_data'] = exam_data

        logging.debug(f"Fair Value Gaps - Stored Chart Data Length: {len(chart_data)}")
        logging.debug(f"Fair Value Gaps - Stored Chart Data Sample (last 5): {chart_data[-5:]}")
        logging.debug(f"Fair Value Gaps - Coin: {coin}, Timeframe: {timeframe}")

        return render_template(
            'fair_value_gaps.html',
            chart_data=chart_data,
            progress={'chart_count': exam_data['chart_count'], 'fvg_part': exam_data['fvg_part']},
            symbol=coin.upper(),
            timeframe=timeframe
        )
    return jsonify({'message': 'Fair Value Gaps POST received'})

@app.route('/charting_exam/orderblocks', methods=['GET', 'POST'])
def orderblocks():
    if 'exam_data' not in session:
        session['exam_data'] = {
            'chart_count': 1,
            'scores': [],
            'chart_data': None,
            'coin': None,
            'timeframe': None
        }

    exam_data = session['exam_data']

    if request.method == 'GET':
        chart_data, coin, timeframe = fetch_chart_data()
        if not chart_data:
            return render_template(
                'orderblocks.html',
                chart_data=[],
                progress={'chart_count': exam_data['chart_count']},
                symbol="ERROR",
                timeframe=timeframe,
                error="Failed to fetch chart data."
            )
        
        exam_data['chart_data'] = chart_data
        exam_data['coin'] = coin
        exam_data['timeframe'] = timeframe
        session['exam_data'] = exam_data

        return render_template(
            'orderblocks.html',
            chart_data=chart_data,
            progress={'chart_count': exam_data['chart_count']},
            symbol=coin.upper(),
            timeframe=timeframe
        )
    return jsonify({'message': 'Orderblocks POST received'})

@app.route('/fetch_new_chart', methods=['GET'])
def fetch_new_chart():
    exam_data = session.get('exam_data', {'chart_count': 1, 'fibonacci_part': 1, 'fvg_part': 1})
    
    current_chart_count = exam_data.get('chart_count', 1)
    
    chart_data, coin, timeframe = fetch_chart_data()
    if not chart_data:
        return jsonify({
            'chart_data': [],
            'chart_count': current_chart_count,
            'fibonacci_part': exam_data.get('fibonacci_part', 1),
            'fvg_part': exam_data.get('fvg_part', 1),
            'symbol': "ERROR",
            'timeframe': timeframe,
            'error': "Failed to fetch chart data."
        })

    if 'fibonacci_part' in exam_data:
        exam_data['fibonacci_part'] = 1
    if 'fvg_part' in exam_data:
        exam_data['fvg_part'] = 1
        
    exam_data['chart_data'] = chart_data
    exam_data['coin'] = coin
    exam_data['timeframe'] = timeframe
    session['exam_data'] = exam_data

    logging.debug(f"Fetch New Chart - Stored Chart Data Length: {len(chart_data)}")
    logging.debug(f"Fetch New Chart - Stored Chart Data Sample: {chart_data[:5]}")
    logging.debug(f"Fetch New Chart - Current chart_count: {exam_data['chart_count']}")

    return jsonify({
        'chart_data': chart_data,
        'chart_count': exam_data['chart_count'],
        'fibonacci_part': exam_data.get('fibonacci_part', 1),
        'fvg_part': exam_data.get('fvg_part', 1),
        'symbol': coin.upper(),
        'timeframe': timeframe
    })

@app.route('/charting_exam/validate', methods=['POST'])
def validate_drawing():
    data = request.get_json()
    exam_type = data.get('examType')
    drawings = data.get('drawings', [])
    chart_count = data.get('chartCount')

    exam_data = session.get('exam_data', {})
    chart_data = exam_data.get('chart_data', [])
    interval = exam_data.get('timeframe', '4h')
    
    if exam_type == 'fibonacci_retracement':
        fibonacci_part = exam_data.get('fibonacci_part', 1)
        chart_count = exam_data.get('chart_count', 1)
        validation_result = validate_fibonacci_retracement(drawings, chart_data, interval, fibonacci_part)
        
        if fibonacci_part == 1:
            exam_data['scores'].append({'uptrend': validation_result['score']})
            exam_data['fibonacci_part'] = 2
            session['exam_data'] = exam_data
            validation_result['next_part'] = 2
        else:
            exam_data['scores'][-1]['downtrend'] = validation_result['score']
            avg_score = (exam_data['scores'][-1]['uptrend'] + validation_result['score']) / 2
            exam_data['scores'][-1]['average'] = avg_score
            
            current_chart_count = exam_data.get('chart_count', 1)
            if current_chart_count < 5:
                exam_data['chart_count'] = current_chart_count + 1
            exam_data['fibonacci_part'] = 1
            
            session['exam_data'] = exam_data
            validation_result['next_part'] = None
            chart_count = exam_data['chart_count']
            
    elif exam_type == 'swing_analysis':
        validation_result = validate_swing_points(drawings, chart_data, interval)
        chart_count = exam_data.get('chart_count', 1)
        
    elif exam_type == 'fair_value_gaps':
        fvg_part = exam_data.get('fvg_part', 1)
        chart_count = exam_data.get('chart_count', 1)
        validation_result = validate_fair_value_gaps(drawings, chart_data, interval, fvg_part)
        
        if fvg_part == 1:
            exam_data['scores'].append({'bullish': validation_result['score']})
            exam_data['fvg_part'] = 2
            session['exam_data'] = exam_data
            validation_result['next_part'] = 2
        else:
            exam_data['scores'][-1]['bearish'] = validation_result['score']
            avg_score = (exam_data['scores'][-1]['bullish'] + validation_result['score']) / 2
            exam_data['scores'][-1]['average'] = avg_score
            
            current_chart_count = exam_data.get('chart_count', 1)
            if current_chart_count < 5:
                exam_data['chart_count'] = current_chart_count + 1
            exam_data['fvg_part'] = 1
            
            session['exam_data'] = exam_data
            validation_result['next_part'] = None
            chart_count = exam_data['chart_count']
    else:
        return jsonify({'success': False, 'message': 'Exam type not implemented yet'})

    if exam_type not in ['fibonacci_retracement', 'fair_value_gaps']:
        exam_data['scores'].append(validation_result['score'])
    session['exam_data'] = exam_data

    response = {
        'success': validation_result['success'],
        'message': validation_result['message'],
        'chart_count': chart_count,
        'feedback': validation_result['feedback'],
        'score': validation_result['score'],
        'totalExpectedPoints': validation_result['totalExpectedPoints'],
        'expected': validation_result.get('expected', {})
    }
    
    if exam_type == 'fibonacci_retracement':
        response['fibonacci_part'] = fibonacci_part
        response['next_part'] = validation_result.get('next_part')
    elif exam_type == 'fair_value_gaps':
        response['fvg_part'] = fvg_part
        response['next_part'] = validation_result.get('next_part')
        
    response['symbol'] = exam_data.get('coin', 'Unknown').upper()
    
    return jsonify(response)

def validate_swing_points(drawings, chart_data, interval):
    if not chart_data or len(chart_data) < 10:
        return {
            'success': False,
            'message': 'Insufficient chart data for validation.',
            'score': 0,
            'feedback': {'correct': [], 'incorrect': [{'advice': 'No chart data available. Please try again with a different chart.'}]},
            'expected': {'highs': [], 'lows': []},
            'totalExpectedPoints': 0
        }

    swing_points = detect_swing_points(chart_data, timeframe=interval)
    highs = swing_points['highs']
    lows = swing_points['lows']
    expected = {'highs': highs, 'lows': lows}

    if len(highs) + len(lows) == 0:
        return {
            'success': False,
            'message': 'No significant swing points detected in this chart.',
            'score': 0,
            'feedback': {'correct': [], 'incorrect': [{'advice': 'This chart does not have any significant swing points. Try another chart.'}]},
            'expected': expected,
            'totalExpectedPoints': 0
        }

    price_range = max(c['high'] for c in chart_data) - min(c['low'] for c in chart_data) if chart_data else 1
    tolerance_map = {'1h': 0.005, '4h': 0.015, '1d': 0.025, '1w': 0.035}
    price_tolerance = price_range * tolerance_map.get(interval, 0.02)
    time_increment = {'1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800}.get(interval, 14400)
    time_tolerance = time_increment * 3

    matched = 0
    feedback = {'correct': [], 'incorrect': []}
    used_points = set()

    for d in drawings:
        point_matched = False
        for i, point in enumerate(highs + lows):
            if i in used_points:
                continue
            if (abs(d['time'] - point['time']) < time_tolerance and
                abs(d['price'] - point['price']) < price_tolerance):
                matched += 1
                point_matched = True
                used_points.add(i)
                point_type = 'high' if point in highs else 'low'
                feedback['correct'].append({
                    'type': point_type,
                    'time': point['time'],
                    'price': point['price'],
                    'advice': f"Good job! You identified a significant swing {point_type} at price {point['price']:.2f}."
                })
                break
        if not point_matched:
            feedback['incorrect'].append({
                'type': d['type'],
                'time': d['time'],
                'price': d['price'],
                'advice': f"This point at price {d['price']:.2f} doesn't match a significant swing point."
            })

    for i, point in enumerate(highs + lows):
        if i not in used_points:
            point_type = 'high' if point in highs else 'low'
            feedback['incorrect'].append({
                'type': 'missed_point',
                'time': point['time'],
                'price': point['price'],
                'advice': f"You missed a significant swing {point_type} at price {point['price']:.2f}."
            })

    total_expected = len(highs) + len(lows)
    success = matched == total_expected
    score = matched

    return {
        'success': success,
        'message': 'All significant swing points identified correctly!' if success else 'Some swing points were missed or incorrect.',
        'score': score,
        'feedback': feedback,
        'totalExpectedPoints': total_expected,
        'expected': expected
    }

def validate_fibonacci_retracement(drawings, chart_data, interval, part):
    default_expected = {'start': {'time': 0, 'price': 0}, 'end': {'time': 0, 'price': 0}, 'direction': 'unknown'}

    logging.debug(f"Validate Fibonacci - Chart Data Length: {len(chart_data)}")
    if not chart_data or len(chart_data) < 10:
        logging.debug("Insufficient chart data in validate_fibonacci_retracement")
        return {
            'success': False,
            'message': 'Insufficient chart data for validation.',
            'score': 0,
            'feedback': {'correct': [], 'incorrect': [{'advice': 'No chart data available. Please try again with a different chart.'}]},
            'totalExpectedPoints': 0,
            'expected': default_expected
        }

    swing_points = detect_swing_points(chart_data, timeframe=interval)
    highs = swing_points['highs']
    lows = swing_points['lows']

    logging.debug(f"Validate Fibonacci - Detected {len(highs)} highs and {len(lows)} lows")

    if len(highs) < 1 or len(lows) < 1:
        logging.debug("Not enough significant swing points for a retracement")
        return {
            'success': False,
            'message': 'Not enough significant swing points for a retracement.',
            'score': 0,
            'feedback': {'correct': [], 'incorrect': [{'advice': 'This chart lacks clear swing points. Try another chart.'}]},
            'totalExpectedPoints': 0,
            'expected': default_expected
        }

    lows_with_subsequent_high = [low for low in lows if any(high['time'] > low['time'] for high in highs)]
    if lows_with_subsequent_high:
        uptrend_low = max(lows_with_subsequent_high, key=lambda x: x['time'])
        subsequent_highs = [high for high in highs if high['time'] > uptrend_low['time']]
        uptrend_high = max(subsequent_highs, key=lambda x: x['time']) if subsequent_highs else None
    else:
        uptrend_low = None
        uptrend_high = None

    highs_with_subsequent_low = [high for high in highs if any(low['time'] > high['time'] for low in lows)]
    if highs_with_subsequent_low:
        downtrend_high = max(highs_with_subsequent_low, key=lambda x: x['time'])
        subsequent_lows = [low for low in lows if low['time'] > downtrend_high['time']]
        downtrend_low = max(subsequent_lows, key=lambda x: x['time']) if subsequent_lows else None
    else:
        downtrend_high = None
        downtrend_low = None

    expected_retracement = (
        {'start': uptrend_low, 'end': uptrend_high, 'direction': 'uptrend'} if part == 1 and uptrend_low and uptrend_high else
        {'start': downtrend_high, 'end': downtrend_low, 'direction': 'downtrend'} if part == 2 and downtrend_high and downtrend_low else
        default_expected
    )

    logging.debug(f"Validate Fibonacci - Expected Retracement: {expected_retracement}")

    if expected_retracement == default_expected:
        logging.debug(f"No significant {'uptrend' if part == 1 else 'downtrend'} retracement found")
        return {
            'success': False,
            'message': f"No significant {'uptrend' if part == 1 else 'downtrend'} retracement found.",
            'score': 0,
            'feedback': {'correct': [], 'incorrect': [{'advice': f"Couldn't find a clear {'uptrend' if part == 1 else 'downtrend'} retracement. Try another chart."}]},
            'totalExpectedPoints': 2,
            'expected': expected_retracement
        }

    price_range = max(c['high'] for c in chart_data) - min(c['low'] for c in chart_data) if chart_data else 1
    tolerance_map = {'1h': 0.01, '4h': 0.02, '1d': 0.03, '1w': 0.04}
    price_tolerance = price_range * tolerance_map.get(interval, 0.02)
    time_increment = {'1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800}.get(interval, 14400)
    time_tolerance = time_increment * 3

    total_credits = 2
    credits_earned = 0
    feedback = {'correct': [], 'incorrect': []}

    if not drawings:
        feedback['incorrect'].append({
            'type': 'missed_retracement',
            'direction': expected_retracement['direction'],
            'startPrice': expected_retracement['start']['price'],
            'endPrice': expected_retracement['end']['price'],
            'advice': f"You missed the {expected_retracement['direction']} retracement from {expected_retracement['start']['price']:.2f} to {expected_retracement['end']['price']:.2f}."
        })
    else:
        for fib in drawings:
            user_direction = 'uptrend' if fib['end']['price'] > fib['start']['price'] else 'downtrend'
            direction_matched = user_direction == expected_retracement['direction']

            if not direction_matched:
                feedback['incorrect'].append({
                    'type': 'incorrect_direction',
                    'direction': user_direction,
                    'startPrice': fib['start']['price'],
                    'endPrice': fib['end']['price'],
                    'advice': f"Direction incorrect: Expected {expected_retracement['direction']}, but you drew a {user_direction} from {fib['start']['price']:.2f} to {fib['end']['price']:.2f}."
                })
                continue

            start_exact = (abs(fib['start']['time'] - expected_retracement['start']['time']) < time_tolerance and
                           abs(fib['start']['price'] - expected_retracement['start']['price']) < price_tolerance)
            start_close = (abs(fib['start']['time'] - expected_retracement['start']['time']) < time_tolerance * 2 and
                           abs(fib['start']['price'] - expected_retracement['start']['price']) < price_tolerance * 2)
            
            start_credits = 1 if start_exact else 0.5 if start_close else 0
            credits_earned += start_credits

            end_exact = (abs(fib['end']['time'] - expected_retracement['end']['time']) < time_tolerance and
                         abs(fib['end']['price'] - expected_retracement['end']['price']) < price_tolerance)
            end_close = (abs(fib['end']['time'] - expected_retracement['end']['time']) < time_tolerance * 2 and
                         abs(fib['end']['price'] - expected_retracement['end']['price']) < price_tolerance * 2)
            
            end_credits = 1 if end_exact else 0.5 if end_close else 0
            credits_earned += end_credits

            feedback['correct'].append({
                'direction': user_direction,
                'startPrice': fib['start']['price'],
                'endPrice': fib['end']['price'],
                'startCredits': start_credits,
                'endCredits': end_credits,
                'advice': f"Start Price: {start_credits}/1 credit ({'Exact' if start_exact else 'Close' if start_close else 'Incorrect'}), End Price: {end_credits}/1 credit ({'Exact' if end_exact else 'Close' if end_close else 'Incorrect'})"
            })

        if credits_earned == 0:
            feedback['incorrect'].append({
                'type': 'missed_retracement',
                'direction': expected_retracement['direction'],
                'startPrice': expected_retracement['start']['price'],
                'endPrice': expected_retracement['end']['price'],
                'advice': f"You missed the {expected_retracement['direction']} retracement from {expected_retracement['start']['price']:.2f} to {expected_retracement['end']['price']:.2f}."
            })

    success = credits_earned > 0
    score = credits_earned

    return {
        'success': success,
        'message': f"{'Uptrend' if part == 1 else 'Downtrend'} retracement: {score}/{total_credits} credits earned!",
        'score': score,
        'feedback': feedback,
        'totalExpectedPoints': total_credits,
        'expected': expected_retracement,
        'next_part': 2 if part == 1 else None
    }

def validate_fair_value_gaps(drawings, chart_data, interval, part):
    """
    Validate user-identified FVGs against detected FVGs using strict pattern matching.
    
    Args:
        drawings: User submitted FVG drawings
        chart_data: Chart candle data
        interval: Timeframe interval
        part: 1 for bullish, 2 for bearish
        
    Returns:
        Validation results
    """
    gap_type = 'bullish' if part == 1 else 'bearish'
    
    # Check if user indicated "No FVGs Found"
    no_fvgs_indicated = len(drawings) == 1 and drawings[0].get('no_fvgs_found', False)
    
    # Detect FVGs using strict pattern definition
    expected_gaps = detect_fair_value_gaps(chart_data, gap_type)
    
    # If user indicated "No FVGs Found" and we didn't find any either, this is correct
    if no_fvgs_indicated and not expected_gaps:
        return {
            'success': True,
            'message': f'Correct! There are no {gap_type} fair value gaps in this chart.',
            'score': 1,
            'feedback': {
                'correct': [{
                    'type': 'no_gaps',
                    'advice': 'Good job! You correctly identified that there are no fair value gaps in this chart.'
                }],
                'incorrect': []
            },
            'totalExpectedPoints': 1,
            'expected': {'gaps': []}
        }
    
    # If user indicated "No FVGs Found" but we found gaps, this is incorrect
    if no_fvgs_indicated and expected_gaps:
        return {
            'success': False,
            'message': f'There are actually {len(expected_gaps)} {gap_type} fair value gaps in this chart.',
            'score': 0,
            'feedback': {
                'correct': [],
                'incorrect': [{
                    'type': 'missed_all_gaps',
                    'advice': f'You indicated no FVGs, but there are {len(expected_gaps)} {gap_type} fair value gaps present.'
                }]
            },
            'totalExpectedPoints': len(expected_gaps),
            'expected': {'gaps': expected_gaps}
        }
    
    # If we didn't find any gaps and user made drawings
    if not expected_gaps and not no_fvgs_indicated:
        return {
            'success': False,
            'message': f'No {gap_type} fair value gaps detected in this chart.',
            'score': 0,
            'feedback': {
                'correct': [],
                'incorrect': [{
                    'type': 'no_gaps',
                    'advice': f'There are no {gap_type} fair value gaps in this chart based on the strict definition. You should have used the "No FVGs Found" button.'
                }]
            },
            'totalExpectedPoints': 0,
            'expected': {'gaps': []}
        }
    
    # Calculate tolerances based on chart range
    price_range = max(c['high'] for c in chart_data) - min(c['low'] for c in chart_data) if chart_data else 1
    
    # More generous tolerance values to account for close user markings
    tolerance_map = {'1h': 0.015, '4h': 0.02, '1d': 0.025, '1w': 0.03}
    price_tolerance = price_range * tolerance_map.get(interval, 0.02)
    
    time_increment = {'1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800}.get(interval, 14400)
    time_tolerance = time_increment * 5  # More generous time tolerance
    
    matched_gaps = []
    feedback = {'correct': [], 'incorrect': []}
    used_gaps = set()
    
    for drawing in drawings:
        drawing_type = drawing.get('type', gap_type)
        if drawing_type != gap_type:
            feedback['incorrect'].append({
                'type': 'incorrect_type',
                'topPrice': drawing.get('topPrice'),
                'bottomPrice': drawing.get('bottomPrice'),
                'advice': f'You marked a {drawing_type} gap, but we\'re looking for {gap_type} gaps in this part.'
            })
            continue
        
        gap_matched = False
        
        for i, gap in enumerate(expected_gaps):
            if i in used_gaps:
                continue
            
            # Check if this is a horizontal line (h-line) drawing
            is_hline = abs(drawing.get('topPrice', 0) - drawing.get('bottomPrice', 0)) < price_tolerance / 10
            
            if is_hline:
                # For h-lines, check if line is within the gap area
                price = drawing.get('topPrice', 0)
                price_in_gap = (
                    price <= gap['topPrice'] + price_tolerance and 
                    price >= gap['bottomPrice'] - price_tolerance
                )
                
                if price_in_gap:
                    matched_gaps.append(gap)
                    used_gaps.add(i)
                    gap_matched = True
                    feedback['correct'].append({
                        'type': gap_type,
                        'topPrice': gap['topPrice'],
                        'bottomPrice': gap['bottomPrice'],
                        'size': gap['size'],
                        'advice': f'Good job! You correctly identified a {gap_type} fair value gap.'
                    })
                    break
            else:
                # For rectangles, check both top and bottom boundaries
                top_match = abs(drawing.get('topPrice', 0) - gap['topPrice']) < price_tolerance
                bottom_match = abs(drawing.get('bottomPrice', 0) - gap['bottomPrice']) < price_tolerance
                
                # Allow some flexibility in matching
                gap_overlap = (
                    min(drawing.get('topPrice', 0), gap['topPrice']) >= 
                    max(drawing.get('bottomPrice', 0), gap['bottomPrice'])
                )
                
                # Check time overlap or proximity
                time_match = (
                    abs(drawing.get('startTime', 0) - gap['startTime']) < time_tolerance or
                    abs(drawing.get('endTime', 0) - gap['endTime']) < time_tolerance or
                    (drawing.get('startTime', 0) <= gap['endTime'] and 
                     drawing.get('endTime', 0) >= gap['startTime'])
                )
                
                if ((top_match or bottom_match or gap_overlap) and time_match):
                    matched_gaps.append(gap)
                    used_gaps.add(i)
                    gap_matched = True
                    feedback['correct'].append({
                        'type': gap_type,
                        'topPrice': gap['topPrice'],
                        'bottomPrice': gap['bottomPrice'],
                        'size': gap['size'],
                        'advice': f'Good job! You correctly identified a {gap_type} fair value gap.'
                    })
                    break
        
        if not gap_matched:
            feedback['incorrect'].append({
                'type': 'incorrect_gap',
                'topPrice': drawing.get('topPrice'),
                'bottomPrice': drawing.get('bottomPrice'),
                'advice': f'This does not match the strict {gap_type} FVG pattern. Remember: NO OVERLAP between 1st and 3rd candles.'
            })
    
    # Add missed gaps to feedback
    for i, gap in enumerate(expected_gaps):
        if i not in used_gaps:
            feedback['incorrect'].append({
                'type': 'missed_gap',
                'topPrice': gap['topPrice'],
                'bottomPrice': gap['bottomPrice'],
                'size': gap['size'],
                'advice': f'You missed a {gap_type} fair value gap from {gap["bottomPrice"]:.2f} to {gap["topPrice"]:.2f}.'
            })
    
    score = len(matched_gaps)
    total_expected = len(expected_gaps)
    success = score == total_expected and score > 0
    
    return {
        'success': success,
        'message': f"{gap_type.capitalize()} Fair Value Gaps: {score}/{total_expected} correctly identified!",
        'score': score,
        'feedback': feedback,
        'totalExpectedPoints': total_expected,
        'expected': {'gaps': expected_gaps},
        'next_part': 2 if part == 1 else None
    }

if __name__ == '__main__':
    app.run(debug=True)