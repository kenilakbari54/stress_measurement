from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model/stress_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def quiz():
    return render_template('quiz.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Extract answers from the form
    answers = [
        int(request.form['question1']),
        int(request.form['question2']),
        int(request.form['question3']),
        int(request.form['question4']),
        int(request.form['question5']),
        int(request.form['question6']),
        int(request.form['question7']),
        int(request.form['question8']),
        int(request.form['question9']),
        int(request.form['question10'])
    ]

    # Convert to numpy array for prediction
    answers_array = np.array(answers).reshape(1, -1)

    # Predict stress level
    stress_level = model.predict(answers_array)[0]

    # Determine if user is stressed or not
    result = "You are stressed." if stress_level > 3 else "You are not stressed."

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
