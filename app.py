from flask import Flask, render_template, request, session, jsonify
import requests
import random
import json
import logging

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

logging.basicConfig(level=logging.DEBUG)

def fetch_chart_data(coin=None, timeframe=None, limit=50):
    # Define available coins and timeframes
    all_coins = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cosmos', 'ripple', 'litecoin', 'chainlink']
    hourly_coins = ['bitcoin', 'ethereum', 'binancecoin', 'solana']
    timeframes = {'1h': 1, '4h': 7, '1d': 30, '1w': 90}
    
    # Select timeframe if not provided
    timeframe = timeframe or random.choice(list(timeframes.keys()))
    
    # Select coin based on timeframe restrictions
    if coin is None:
        if timeframe == '1h':
            coin = random.choice(hourly_coins)
        else:
            coin = random.choice(all_coins)
    
    days = timeframes[timeframe]
    url = f'https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency=usd&days={days}'
    headers = {"x-cg-demo-api-key": "CG-X9rKSiVeFyMS6FPbUCaFw4Lc"}
    print(f"Fetching data for {coin} ({timeframe}) from {url} with days={days}")
    logging.info(f"Selected coin: {coin} for timeframe: {timeframe}")
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        print(f"API Response Status: {response.status_code}")
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            dummy_data = [
                {'time': 1710960000, 'open': 0.5, 'high': 0.51, 'low': 0.49, 'close': 0.505},
                {'time': 1710963600, 'open': 0.505, 'high': 0.515, 'low': 0.5, 'close': 0.51}
            ]
            return dummy_data, coin, timeframe
        
        data = response.json()
        print(f"Received {len(data)} candles from CoinGecko")
        candles_to_take = min(limit, len(data))
        chart_data = []
        
        for row in data[-candles_to_take:]:
            chart_data.append({
                'time': int(row[0] / 1000), 
                'open': float(row[1]), 
                'high': float(row[2]), 
                'low': float(row[3]), 
                'close': float(row[4])
            })
        
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
    lookback_map = {'1h': 8, '4h': 5, '1d': 3, '1w': 2}  # Added weekly timeframe
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
    # Always clear exam data when visiting the main exams page
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

@app.route('/charting_exam/fibonacci_retracement', methods=['GET', 'POST'])
def fibonacci_retracement():
    # Check if we need to reset exam data
    if request.args.get('reset') == 'true':
        session.pop('exam_data', None)

    # Initialize exam data if not present
    if 'exam_data' not in session:
        session['exam_data'] = {
            'chart_count': 1,
            'fibonacci_part': 1,  # 1 = uptrend, 2 = downtrend
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
        exam_data['fibonacci_part'] = 1  # Start with uptrend
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

@app.route('/fetch_new_chart', methods=['GET'])
def fetch_new_chart():
    exam_data = session.get('exam_data', {'chart_count': 1, 'fibonacci_part': 1})
    
    # Don't change the chart count here - it was already incremented in the validation step
    current_chart_count = exam_data.get('chart_count', 1)
    
    chart_data, coin, timeframe = fetch_chart_data()
    if not chart_data:
        chart_data = [
            {'time': 1710960000, 'open': 0.5, 'high': 0.51, 'low': 0.49, 'close': 0.505},
            {'time': 1710963600, 'open': 0.505, 'high': 0.515, 'low': 0.5, 'close': 0.51}
        ]

    # Only update these properties, don't increment the chart count here
    exam_data['fibonacci_part'] = 1  # Reset to uptrend for new chart
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
        'fibonacci_part': exam_data['fibonacci_part'],
        'symbol': coin.upper(),
        'timeframe': timeframe
    })

@app.route('/charting_exam/validate', methods=['POST'])
def validate_drawing():
    data = request.get_json()
    exam_type = data.get('examType')
    drawings = data.get('drawings', [])
    chart_count = data.get('chartCount')  # Frontend sends this, but we'll override with session

    exam_data = session.get('exam_data', {})
    chart_data = exam_data.get('chart_data', [])
    interval = exam_data.get('timeframe', '4h')
    fibonacci_part = exam_data.get('fibonacci_part', 1) if exam_type == 'fibonacci_retracement' else None
    chart_count = exam_data.get('chart_count', 1)  # Use session value for consistency

    logging.debug(f"Validate Drawing - Chart Count: {chart_count}, Part: {fibonacci_part}")
    logging.debug(f"Validate Drawing - Chart Data Length: {len(chart_data)}")
    logging.debug(f"Validate Drawing - Chart Data Sample: {chart_data[:5]}")

    if exam_type == 'swing_analysis':
        validation_result = validate_swing_points(drawings, chart_data, interval)
    elif exam_type == 'fibonacci_retracement':
        validation_result = validate_fibonacci_retracement(drawings, chart_data, interval, fibonacci_part)
        
        if fibonacci_part == 1:
            # Store uptrend score, switch to downtrend
            exam_data['scores'].append({'uptrend': validation_result['score']})
            exam_data['fibonacci_part'] = 2
            session['exam_data'] = exam_data
            validation_result['next_part'] = 2
        else:
            # Store downtrend score, average, and move on
            exam_data['scores'][-1]['downtrend'] = validation_result['score']
            avg_score = (exam_data['scores'][-1]['uptrend'] + validation_result['score']) / 2
            exam_data['scores'][-1]['average'] = avg_score
            
            # Increment chart_count after completing both parts, but don't exceed maxCharts
            current_chart_count = exam_data.get('chart_count', 1)
            if current_chart_count < 5:  # Ensure we don't exceed the maximum
                exam_data['chart_count'] = current_chart_count + 1
            exam_data['fibonacci_part'] = 1  # Reset for next chart
            
            session['exam_data'] = exam_data
            validation_result['next_part'] = None  # Signal completion of the chart
            chart_count = exam_data['chart_count']  # Update chart_count for response
    else:
        return jsonify({'success': False, 'message': 'Exam type not implemented yet'})

    if exam_type != 'fibonacci_retracement':
        exam_data['scores'].append(validation_result['score'])
    session['exam_data'] = exam_data

    return jsonify({
        'success': validation_result['success'],
        'message': validation_result['message'],
        'chart_count': chart_count,
        'fibonacci_part': fibonacci_part if exam_type == 'fibonacci_retracement' else None,
        'next_part': validation_result.get('next_part') if exam_type == 'fibonacci_retracement' else None,
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
    tolerance_map = {'1h': 0.005, '4h': 0.015, '1d': 0.025, '1w': 0.035}  # Added weekly timeframe
    price_tolerance = price_range * tolerance_map.get(interval, 0.02)
    time_increment = {'1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800}.get(interval, 14400)  # Added weekly
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

    # Check if there's enough chart data
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

    # Detect swing points
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

    # For uptrend (part 1): Find the most recent swing low with a subsequent swing high
    lows_with_subsequent_high = [low for low in lows if any(high['time'] > low['time'] for high in highs)]
    if lows_with_subsequent_high:
        uptrend_low = max(lows_with_subsequent_high, key=lambda x: x['time'])
        subsequent_highs = [high for high in highs if high['time'] > uptrend_low['time']]
        uptrend_high = max(subsequent_highs, key=lambda x: x['time']) if subsequent_highs else None
    else:
        uptrend_low = None
        uptrend_high = None

    # For downtrend (part 2): Find the most recent swing high with a subsequent swing low
    highs_with_subsequent_low = [high for high in highs if any(low['time'] > high['time'] for low in lows)]
    if highs_with_subsequent_low:
        downtrend_high = max(highs_with_subsequent_low, key=lambda x: x['time'])
        subsequent_lows = [low for low in lows if low['time'] > downtrend_high['time']]
        downtrend_low = max(subsequent_lows, key=lambda x: x['time']) if subsequent_lows else None
    else:
        downtrend_high = None
        downtrend_low = None

    # Determine expected retracement based on the part
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
            'totalExpectedPoints': 2,  # Still expect 2 credits even if no retracement is found
            'expected': expected_retracement
        }

    # Define tolerances for validation
    price_range = max(c['high'] for c in chart_data) - min(c['low'] for c in chart_data) if chart_data else 1
    tolerance_map = {'1h': 0.01, '4h': 0.02, '1d': 0.03, '1w': 0.04}  # Added weekly
    price_tolerance = price_range * tolerance_map.get(interval, 0.02)
    time_increment = {'1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800}.get(interval, 14400)  # Added weekly
    time_tolerance = time_increment * 3

    total_credits = 2  # 1 for start, 1 for end
    credits_earned = 0
    feedback = {'correct': [], 'incorrect': []}

    # Validate user drawings
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
                continue  # No credits if direction is wrong

            # Check start price
            start_exact = (abs(fib['start']['time'] - expected_retracement['start']['time']) < time_tolerance and
                           abs(fib['start']['price'] - expected_retracement['start']['price']) < price_tolerance)
            start_close = (abs(fib['start']['time'] - expected_retracement['start']['time']) < time_tolerance * 2 and
                           abs(fib['start']['price'] - expected_retracement['start']['price']) < price_tolerance * 2)
            
            start_credits = 1 if start_exact else 0.5 if start_close else 0
            credits_earned += start_credits

            # Check end price
            end_exact = (abs(fib['end']['time'] - expected_retracement['end']['time']) < time_tolerance and
                         abs(fib['end']['price'] - expected_retracement['end']['price']) < price_tolerance)
            end_close = (abs(fib['end']['time'] - expected_retracement['end']['time']) < time_tolerance * 2 and
                         abs(fib['end']['price'] - expected_retracement['end']['price']) < price_tolerance * 2)
            
            end_credits = 1 if end_exact else 0.5 if end_close else 0
            credits_earned += end_credits

            # Provide feedback for this retracement
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
        'next_part': 2 if part == 1 else None  # Indicate next part or end of chart
    }

if __name__ == '__main__':
    app.run(debug=True)