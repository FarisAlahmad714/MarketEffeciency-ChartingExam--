from flask import Flask, render_template, request, session, jsonify
import requests
import random
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

def fetch_chart_data(coin=None, timeframe=None, limit=50):
    coins = [
        'bitcoin', 'ethereum', 'binancecoin', 'solana', 'matic-network', 'polkadot',
        'cosmos', 'tezos', 'near', 'ripple', 'litecoin', 'chainlink'
    ]
    timeframes = {'1h': 1, '4h': 7, '1d': 30}
    
    coin = coin or random.choice(coins)
    timeframe = timeframe or random.choice(list(timeframes.keys()))
    
    days = timeframes[timeframe]
    url = f'https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency=usd&days={days}'
    headers = {"x-cg-demo-api-key": "CG-X9rKSiVeFyMS6FPbUCaFw4Lc"}
    print(f"Fetching data for {coin} ({timeframe}) from {url} with days={days}")
    try:
        response = requests.get(url, headers=headers, timeout=5)
        print(f"API Response Status: {response.status_code}")
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return [], coin, timeframe
        data = response.json()
        print(f"Received {len(data)} candles from CoinGecko")
        candles_to_take = min(limit, len(data))
        chart_data = [
            {'time': int(row[0] / 1000), 'open': float(row[1]), 'high': float(row[2]), 
             'low': float(row[3]), 'close': float(row[4])}
            for row in data[-candles_to_take:]
        ]
        print(f"Fetched {len(chart_data)} candles for {coin} ({timeframe})")
        return chart_data, coin, timeframe
    except Exception as e:
        print(f"Exception fetching data: {e}")
        dummy_data = [
            {'time': 1710960000, 'open': 0.5, 'high': 0.51, 'low': 0.49, 'close': 0.505},
            {'time': 1710963600, 'open': 0.505, 'high': 0.515, 'low': 0.5, 'close': 0.51}
        ]
        print(f"Using dummy data: {len(dummy_data)} candles")
        return dummy_data, coin, timeframe

def detect_swing_points(data, lookback=5, timeframe='4h', significance_threshold=0.01):
    lookback_map = {'1h': 8, '4h': 5, '1d': 3}
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
        chart_data = [
            {'time': 1710960000, 'open': 0.5, 'high': 0.51, 'low': 0.49, 'close': 0.505},
            {'time': 1710963600, 'open': 0.505, 'high': 0.515, 'low': 0.5, 'close': 0.51}
        ]
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

@app.route('/fetch_new_chart', methods=['GET'])
def fetch_new_chart():
    exam_data = session.get('exam_data', {'chart_count': 1})
    chart_data, coin, timeframe = fetch_chart_data()
    if not chart_data:
        chart_data = [
            {'time': 1710960000, 'open': 0.5, 'high': 0.51, 'low': 0.49, 'close': 0.505},
            {'time': 1710963600, 'open': 0.505, 'high': 0.515, 'low': 0.5, 'close': 0.51}
        ]

    exam_data['chart_count'] = exam_data.get('chart_count', 1) + 1
    exam_data['chart_data'] = chart_data
    exam_data['coin'] = coin
    exam_data['timeframe'] = timeframe
    session['exam_data'] = exam_data

    return jsonify({
        'chart_data': chart_data,
        'chart_count': exam_data['chart_count'],
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

    if exam_type == 'swing_analysis':
        validation_result = validate_swing_points(drawings, chart_data, interval)
    elif exam_type == 'fibonacci_retracement':
        validation_result = validate_fibonacci_retracement(drawings, chart_data, interval)
    else:
        return jsonify({'success': False, 'message': 'Exam type not implemented yet'})

    exam_data['scores'].append(validation_result['score'])
    session['exam_data'] = exam_data

    return jsonify({
        'success': validation_result['success'],
        'message': validation_result['message'],
        'chart_count': chart_count,
        'symbol': exam_data.get('coin', 'Unknown').upper(),
        'feedback': validation_result['feedback'],
        'score': validation_result['score'],
        'totalExpectedPoints': validation_result['totalExpectedPoints'],
        'expected': validation_result.get('expected', {})
    })

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
    tolerance_map = {'1h': 0.005, '4h': 0.015, '1d': 0.025}
    price_tolerance = price_range * tolerance_map.get(interval, 0.02)
    time_increment = {'1h': 3600, '4h': 14400, '1d': 86400}.get(interval, 14400)
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
                'advice': f"This point at price {d['price']:.2f} doesn’t match a significant swing point."
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

def validate_fibonacci_retracement(drawings, chart_data, interval):
    default_expected = {'start': {'time': 0, 'price': 0}, 'end': {'time': 0, 'price': 0}, 'direction': 'unknown'}

    if not chart_data or len(chart_data) < 10:
        return {
            'success': False,
            'message': 'Insufficient chart data for validation.',
            'score': 0,
            'feedback': {'correct': [], 'incorrect': [{'advice': 'No chart data available. Please try again with a different chart.'}]},
            'totalExpectedPoints': 0,
            'expected': default_expected
        }

    swing_points = detect_swing_points(chart_data, timeframe=interval)
    highs = sorted(swing_points['highs'], key=lambda x: x['price'], reverse=True)
    lows = sorted(swing_points['lows'], key=lambda x: x['price'])

    if len(highs) < 1 or len(lows) < 1:
        return {
            'success': False,
            'message': 'Not enough significant swing points to form a Fibonacci retracement.',
            'score': 0,
            'feedback': {'correct': [], 'incorrect': [{'advice': 'This chart does not have enough significant swing points for a Fibonacci retracement. Try another chart.'}]},
            'totalExpectedPoints': 0,
            'expected': default_expected
        }

    trend = determine_trend(chart_data)
    print(f"Detected trend: {trend}")

    if trend == 'uptrend':
        swing_high = max(highs, key=lambda x: x['time'])
        swing_low = max([low for low in lows if low['time'] < swing_high['time']], key=lambda x: x['time'], default=None)
        expected_direction = 'uptrend'
    elif trend == 'downtrend':
        swing_low = min(lows, key=lambda x: x['time'])
        swing_high = min([high for high in highs if high['time'] > swing_low['time']], key=lambda x: x['time'], default=None)
        expected_direction = 'downtrend'
    else:
        swing_high = highs[0]
        swing_low = lows[0]
        expected_direction = 'uptrend' if swing_low['time'] < swing_high['time'] else 'downtrend'

    if not swing_high or not swing_low:
        return {
            'success': False,
            'message': 'Could not determine a significant retracement in this chart.',
            'score': 0,
            'feedback': {'correct': [], 'incorrect': [{'advice': 'This chart does not have a clear retracement opportunity. Try another chart.'}]},
            'totalExpectedPoints': 0,
            'expected': default_expected
        }

    expected_retracement = {
        'start': swing_low if expected_direction == 'uptrend' else swing_high,
        'end': swing_high if expected_direction == 'uptrend' else swing_low,
        'direction': expected_direction
    }

    price_range = max(c['high'] for c in chart_data) - min(c['low'] for c in chart_data) if chart_data else 1
    tolerance_map = {'1h': 0.01, '4h': 0.02, '1d': 0.03}
    price_tolerance = price_range * tolerance_map.get(interval, 0.02)
    time_increment = {'1h': 3600, '4h': 14400, '1d': 86400}.get(interval, 14400)
    time_tolerance = time_increment * 3

    matched = 0
    feedback = {'correct': [], 'incorrect': []}

    for fib in drawings:
        start_matched = (abs(fib['start']['time'] - expected_retracement['start']['time']) < time_tolerance and
                         abs(fib['start']['price'] - expected_retracement['start']['price']) < price_tolerance)
        end_matched = (abs(fib['end']['time'] - expected_retracement['end']['time']) < time_tolerance and
                       abs(fib['end']['price'] - expected_retracement['end']['price']) < price_tolerance)
        
        user_direction = 'uptrend' if fib['end']['price'] > fib['start']['price'] else 'downtrend'
        direction_matched = user_direction == expected_direction

        if start_matched and end_matched and direction_matched:
            matched += 1
            feedback['correct'].append({
                'direction': user_direction,
                'startPrice': fib['start']['price'],
                'endPrice': fib['end']['price'],
                'advice': f"Good job! You correctly drew a Fibonacci retracement for an {user_direction} from {fib['start']['price']:.2f} to {fib['end']['price']:.2f}."
            })
        else:
            feedback['incorrect'].append({
                'type': 'incorrect_retracement',
                'direction': user_direction,
                'startPrice': fib['start']['price'],
                'endPrice': fib['end']['price'],
                'advice': f"This retracement from {fib['start']['price']:.2f} to {fib['end']['price']:.2f} does not match the expected {expected_direction} retracement."
            })

    if matched == 0:
        feedback['incorrect'].append({
            'type': 'missed_retracement',
            'direction': expected_direction,
            'startPrice': expected_retracement['start']['price'],
            'endPrice': expected_retracement['end']['price'],
            'advice': f"You missed the significant {expected_direction} retracement from {expected_retracement['start']['price']:.2f} to {expected_retracement['end']['price']:.2f}."
        })

    total_expected = 1
    success = matched == total_expected
    score = matched

    return {
        'success': success,
        'message': 'Fibonacci retracement identified correctly!' if success else 'The retracement was incorrect or missed.',
        'score': score,
        'feedback': feedback,
        'totalExpectedPoints': total_expected,
        'expected': expected_retracement
    }

@app.route('/charting_exam/fibonacci_retracement', methods=['GET', 'POST'])
def fibonacci_retracement():
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
            chart_data = [
                {'time': 1710960000, 'open': 0.5, 'high': 0.51, 'low': 0.49, 'close': 0.505},
                {'time': 1710963600, 'open': 0.505, 'high': 0.515, 'low': 0.5, 'close': 0.51}
            ]
        exam_data['chart_data'] = chart_data
        exam_data['coin'] = coin
        exam_data['timeframe'] = timeframe
        session['exam_data'] = exam_data

        return render_template(
            'fibonacci_retracement.html',
            chart_data=chart_data,
            progress={'chart_count': exam_data['chart_count']},
            symbol=coin.upper(),
            timeframe=timeframe
        )
    return jsonify({'message': 'Fibonacci Retracement POST received'})

@app.route('/charting_exam/fair_value_gaps', methods=['GET', 'POST'])
def fair_value_gaps():
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
            chart_data = [
                {'time': 1710960000, 'open': 0.5, 'high': 0.51, 'low': 0.49, 'close': 0.505},
                {'time': 1710963600, 'open': 0.505, 'high': 0.515, 'low': 0.5, 'close': 0.51}
            ]
        exam_data['chart_data'] = chart_data
        exam_data['coin'] = coin
        exam_data['timeframe'] = timeframe
        session['exam_data'] = exam_data

        return render_template(
            'fair_value_gaps.html',
            chart_data=chart_data,
            progress={'chart_count': exam_data['chart_count']},
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
            chart_data = [
                {'time': 1710960000, 'open': 0.5, 'high': 0.51, 'low': 0.49, 'close': 0.505},
                {'time': 1710963600, 'open': 0.505, 'high': 0.515, 'low': 0.5, 'close': 0.51}
            ]
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

if __name__ == '__main__':
    app.run(debug=True)