import flask 
from flask import Flask, render_template, request
import joblib
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction' , methods = ['GET' , 'POST'])
def prediction():
    if request.method == 'POST':
        rev = request.form.get('reviews')

        data_point = [rev]

        model = joblib.load('naive_bayes.pkl')

        prediction = model.predict(data_point)

        if prediction[0] == 1:
            result = "😄 Positive Review 😄"
        else:
            result = "☹️ Negative Review ☹️"

        return render_template('output.html', prediction=result)

    return render_template('output.html', prediction="")

if __name__ == '__main__':
    app.run(debug=True)