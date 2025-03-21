from flask import Flask, render_template, request, session, jsonify
import requests
import random
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

def fetch_chart_data(coin=None, timeframe=None, limit=50):
    # Expanded list of PoS assets (and BTC for trading analysis)
    coins = [
        'bitcoin', 'ethereum', 'cardano', 'binancecoin', 'polygon', 'solana',
        'polkadot', 'cosmos', 'tezos', 'near-protocol'
    ]
    timeframes = {'1h': 1, '4h': 7, '1d': 30}
    
    # Randomize coin if not specified
    coin = coin or random.choice(coins)
    # Randomize timeframe if not specified
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
    # Adjust lookback based on timeframe
    lookback_map = {'1h': 8, '4h': 5, '1d': 3}
    lookback = lookback_map.get(timeframe, 5)

    swing_points = {'highs': [], 'lows': []}
    # Calculate the total price range of the chart
    price_range = max(c['high'] for c in data) - min(c['low'] for c in data) if data else 1
    min_price_diff = price_range * significance_threshold  # Minimum price difference for significance

    for i in range(lookback, len(data) - lookback):
        current = data[i]
        # Check for swing high
        before = [c['high'] for c in data[i - lookback:i]]
        after = [c['high'] for c in data[i + 1:i + 1 + lookback]]
        if current['high'] > max(before) and current['high'] > max(after):
            # Check significance: price difference between the high and the lowest low in the window
            window_lows = [c['low'] for c in data[i - lookback:i + 1 + lookback]]
            lowest_low = min(window_lows)
            price_diff = current['high'] - lowest_low
            if price_diff >= min_price_diff:
                swing_points['highs'].append({'time': current['time'], 'price': current['high']})

        # Check for swing low
        before_lows = [c['low'] for c in data[i - lookback:i]]
        after_lows = [c['low'] for c in data[i + 1:i + 1 + lookback]]
        if current['low'] < min(before_lows) and current['low'] < min(after_lows):
            # Check significance: price difference between the low and the highest high in the window
            window_highs = [c['high'] for c in data[i - lookback:i + 1 + lookback]]
            highest_high = max(window_highs)
            price_diff = highest_high - current['low']
            if price_diff >= min_price_diff:
                swing_points['lows'].append({'time': current['time'], 'price': current['low']})

    print(f"Detected {len(swing_points['highs'])} swing highs and {len(swing_points['lows'])} swing lows")
    return swing_points

@app.route('/charting_exams')
def charting_exams():
    # Reset the exam data when returning to the exams page
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
            'timeframe': None  # Will be randomized
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
        'expected': validation_result['expected']
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
    highs = swing_points['highs']  # Use all detected swing highs
    lows = swing_points['lows']    # Use all detected swing lows
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
    success = matched == total_expected  # Success if all significant points are identified
    score = matched

    return {
        'success': success,
        'message': 'All significant swing points identified correctly!' if success else 'Some swing points were missed or incorrect.',
        'score': score,
        'feedback': feedback,
        'totalExpectedPoints': total_expected,
        'expected': expected
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