# app.py
from flask import Flask, render_template

app = Flask(__name__)

# --- Results Data (Copy from your simulation output) ---
# It's often better to load this from a file (e.g., JSON, CSV) in a real app,
# but for simplicity, we'll hardcode it here.
results_data = {
    "1-Bit": 0.4840,
    "2-Bit": 0.4290,
    "2-Level": 0.6480,
    "Tournament": 0.5900,
    "BranchNet_CNN": 0.8283,
    "BranchNet_LSTM": 0.9146,
    "BranchNet_Transformer": 0.5437
}
# You could add more metrics here if you calculate them, e.g., simulation time

@app.route('/') # Define the route for the homepage
def index():
    # Pass the results data to the HTML template
    return render_template('index.html', results=results_data)

if __name__ == '__main__':
    # Make sure debug=False for production
    # host='0.0.0.0' makes it accessible on your network (use with caution)
    app.run(debug=True, host='127.0.0.1', port=5000)