from flask import Flask, render_template, request, session, jsonify
import requests
import random
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

def fetch_chart_data(coin=None, timeframe='1d', limit=50):
    coins = ['bitcoin', 'ethereum', 'binancecoin', 'cardano']
    timeframes = {'1h': 1, '4h': 7, '1d': 30}
    coin = coin or random.choice(coins)
    if timeframe not in timeframes:
        print(f"Invalid timeframe '{timeframe}', defaulting to '1d'")
        timeframe = '1d'
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

def detect_swing_points(data, lookback=5):
    swing_points = {'highs': [], 'lows': []}
    for i in range(lookback, len(data) - lookback):
        current = data[i]
        before = [c['high'] for c in data[i - lookback:i]]
        after = [c['high'] for c in data[i + 1:i + 1 + lookback]]
        if current['high'] > max(before) and current['high'] > max(after):
            swing_points['highs'].append({'time': current['time'], 'price': current['high']})
        before_lows = [c['low'] for c in data[i - lookback:i]]
        after_lows = [c['low'] for c in data[i + 1:i + 1 + lookback]]
        if current['low'] < min(before_lows) and current['low'] < min(after_lows):
            swing_points['lows'].append({'time': current['time'], 'price': current['low']})
    return swing_points

@app.route('/charting_exams')
def charting_exams():
    return render_template('index.html')

@app.route('/charting_exam/swing_analysis', methods=['GET', 'POST'])
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

    if request.method == 'GET':
        chart_data, coin, timeframe = fetch_chart_data(timeframe='4h')
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
    chart_data, coin, timeframe = fetch_chart_data(timeframe='4h')
    if not chart_data:
        chart_data = [
            {'time': 1710960000, 'open': 0.5, 'high': 0.51, 'low': 0.49, 'close': 0.505},
            {'time': 1710963600, 'open': 0.505, 'high': 0.515, 'low': 0.5, 'close': 0.51}
        ]

    exam_data = session.get('exam_data', {'chart_count': 1})
    exam_data['chart_count'] = exam_data.get('chart_count', 1) + 1
    if exam_data['chart_count'] > 5:
        exam_data['chart_count'] = 1
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
        # Define lookback here and pass it to validate_swing_points
        lookback = 5  # Match the default value used in detect_swing_points
        validation_result = validate_swing_points(drawings, chart_data, interval, lookback)
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

def validate_swing_points(drawings, chart_data, interval, lookback):
    if not chart_data or len(chart_data) < 10:
        return {
            'success': False,
            'message': 'Insufficient chart data for validation.',
            'score': 0,
            'feedback': {'correct': [], 'incorrect': [{'advice': 'No chart data available. Please try again with a different chart.'}]},
            'expected': {'highs': [], 'lows': []},
            'totalExpectedPoints': 4
        }

    swing_points = detect_swing_points(chart_data, lookback)
    highs = sorted(swing_points['highs'], key=lambda x: x['price'], reverse=True)[:2]
    lows = sorted(swing_points['lows'], key=lambda x: x['price'])[:2]
    expected = {'highs': highs, 'lows': lows}

    if len(highs) < 2 or len(lows) < 2:
        return {
            'success': False,
            'message': 'Not enough swing points detected in this chart.',
            'score': 0,
            'feedback': {'correct': [], 'incorrect': [{'advice': 'This chart does not have enough clear swing points. Try another chart.'}]},
            'expected': expected,
            'totalExpectedPoints': 4
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
                    'advice': f"Good job! You identified a swing {point_type} at price {point['price']:.2f}. This is a significant {point_type} point because it’s higher/lower than the surrounding {lookback} candles on both sides."
                })
                break
        if not point_matched:
            feedback['incorrect'].append({
                'type': d['type'],
                'time': d['time'],
                'price': d['price'],
                'advice': f"This point at price {d['price']:.2f} doesn’t match a significant swing point. A swing {d['type']} should be higher/lower than the surrounding {lookback} candles on both sides."
            })

    for i, point in enumerate(highs + lows):
        if i not in used_points:
            point_type = 'high' if point in highs else 'low'
            feedback['incorrect'].append({
                'type': 'missed_point',
                'time': point['time'],
                'price': point['price'],
                'advice': f"You missed a swing {point_type} at price {point['price']:.2f}. This is a significant {point_type} because it’s higher/lower than the surrounding {lookback} candles on both sides."
            })

    total_expected = len(highs) + len(lows)
    success = matched >= total_expected
    score = matched

    return {
        'success': success,
        'message': 'Swing points identified correctly!' if success else 'Some swing points were missed or incorrect.',
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
        chart_data, coin, timeframe = fetch_chart_data(timeframe='4h')
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
        chart_data, coin, timeframe = fetch_chart_data(timeframe='4h')
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
        chart_data, coin, timeframe = fetch_chart_data(timeframe='4h')
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