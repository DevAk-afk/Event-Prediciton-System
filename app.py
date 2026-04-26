from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load model and encoders
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

with open(os.path.join(MODEL_DIR, 'attendance_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'label_encoders.pkl'), 'rb') as f:
    le_dict = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'feature_cols.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)

# In-memory storage for events
events_db = []

EVENT_TYPES = ['Cultural Show', 'Technical Workshop', 'Sports', 'Seminar',
               'Quiz Competition', 'Dance Competition', 'Music Night', 'Hackathon']
VENUES = ['Main Auditorium', 'Seminar Hall A', 'Seminar Hall B', 'Outdoor Ground', 'Conference Room']
VENUE_CAPS = {'Main Auditorium': 600, 'Seminar Hall A': 200, 'Seminar Hall B': 150,
              'Outdoor Ground': 800, 'Conference Room': 100}
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
WEATHER = ['Sunny', 'Cloudy', 'Rainy', 'Windy']
TIME_SLOTS = ['Morning (9-12)', 'Afternoon (12-4)', 'Evening (4-8)']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/options', methods=['GET'])
def get_options():
    return jsonify({
        'event_types': EVENT_TYPES,
        'venues': VENUES,
        'venue_caps': VENUE_CAPS,
        'days': DAYS,
        'weather': WEATHER,
        'time_slots': TIME_SLOTS
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        def safe_encode(col, val):
            le = le_dict[col]
            if val in le.classes_:
                return le.transform([val])[0]
            return 0

        cat_cols = ['event_type', 'venue', 'day_of_week', 'time_slot', 'weather']
        encoded = [safe_encode(col, data[col]) for col in cat_cols]

        num_features = [
            int(data['month']),
            int(data['registration_count']),
            int(data['social_media_posts']),
            int(data['is_holiday']),
            int(data['past_event_attendance']),
            int(data['entry_fee']),
            int(data['num_speakers'])
        ]

        features = np.array([encoded + num_features])
        prediction = int(model.predict(features)[0])
        venue_cap = VENUE_CAPS.get(data['venue'], 300)
        prediction = max(5, min(prediction, venue_cap))

        # Confidence band
        confidence_low = max(5, int(prediction * 0.85))
        confidence_high = min(venue_cap, int(prediction * 1.15))

        # Occupancy %
        occupancy = round((prediction / venue_cap) * 100, 1)

        # Key factors
        factors = []
        if int(data['registration_count']) > 150:
            factors.append({'label': 'High Registrations', 'impact': 'positive'})
        if int(data['social_media_posts']) > 25:
            factors.append({'label': 'Strong Social Media Buzz', 'impact': 'positive'})
        if data['weather'] == 'Rainy':
            factors.append({'label': 'Rainy Weather', 'impact': 'negative'})
        if int(data['is_holiday']) == 1:
            factors.append({'label': 'Holiday Boost', 'impact': 'positive'})
        if data['day_of_week'] in ['Saturday', 'Sunday']:
            factors.append({'label': 'Weekend Event', 'impact': 'positive'})
        if data['time_slot'] == 'Evening (4-8)':
            factors.append({'label': 'Prime Time Slot', 'impact': 'positive'})
        if int(data['entry_fee']) > 100:
            factors.append({'label': 'High Entry Fee', 'impact': 'negative'})

        result = {
            'predicted_attendance': prediction,
            'confidence_low': confidence_low,
            'confidence_high': confidence_high,
            'venue_capacity': venue_cap,
            'occupancy_percent': occupancy,
            'factors': factors,
            'recommendation': get_recommendation(occupancy)
        }

        # Save event to history
        events_db.append({
            'id': len(events_db) + 1,
            'event_name': data.get('event_name', 'Unnamed Event'),
            'event_type': data['event_type'],
            'venue': data['venue'],
            'day': data['day_of_week'],
            'month': data['month'],
            'predicted': prediction,
            'occupancy': occupancy,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        })

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_recommendation(occupancy):
    if occupancy >= 90:
        return {'status': 'danger', 'text': 'Near full capacity! Consider a larger venue or limiting registrations.'}
    elif occupancy >= 70:
        return {'status': 'success', 'text': 'Excellent turnout expected! Prepare for a packed event.'}
    elif occupancy >= 50:
        return {'status': 'info', 'text': 'Good attendance predicted. Consider boosting social media promotion.'}
    elif occupancy >= 30:
        return {'status': 'warning', 'text': 'Moderate attendance. Increase outreach and registration drives.'}
    else:
        return {'status': 'danger', 'text': 'Low attendance predicted. Reconsider timing, venue, or promotion strategy.'}

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(events_db[-20:][::-1])

@app.route('/api/stats', methods=['GET'])
def get_stats():
    if not events_db:
        return jsonify({'total_events': 0, 'avg_predicted': 0, 'avg_occupancy': 0, 'top_event_type': 'N/A'})
    
    total = len(events_db)
    avg_pred = round(sum(e['predicted'] for e in events_db) / total)
    avg_occ = round(sum(e['occupancy'] for e in events_db) / total, 1)
    type_counts = {}
    for e in events_db:
        type_counts[e['event_type']] = type_counts.get(e['event_type'], 0) + 1
    top_type = max(type_counts, key=type_counts.get) if type_counts else 'N/A'

    return jsonify({
        'total_events': total,
        'avg_predicted': avg_pred,
        'avg_occupancy': avg_occ,
        'top_event_type': top_type
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
