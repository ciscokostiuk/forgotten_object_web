from flask import Flask, render_template, request, redirect, url_for
import json
import threading
from detector import ForgottenObjectDetector

app = Flask(__name__)
CONFIG_PATH = 'config.json'
detector_thread = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        config = {
            "min_area": int(request.form['min_area']),
            "forgotten_time": int(request.form['forgotten_time']),
            "video_source": int(request.form['video_source']),
            "target_objects": [x.strip() for x in request.form['target_objects'].split(',')],
            "email_notify": request.form.get('email_notify', ''),
            "telegram_token": request.form.get('telegram_token', ''),
            "telegram_chat_id": request.form.get('telegram_chat_id', '')
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)

        global detector_thread
        if detector_thread is None or not detector_thread.is_alive():
            detector = ForgottenObjectDetector(CONFIG_PATH)
            detector_thread = threading.Thread(target=detector.run, daemon=True)
            detector_thread.start()

        return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)