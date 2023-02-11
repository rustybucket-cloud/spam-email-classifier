from flask import Flask, request, jsonify, render_template
from classify import classify
from train import SpamModel
import logging
app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def main():
    # API requests
    if request.method == "POST":
        logging.info("INFO:POST request to Spam Predictor Model")

        # if the request came from a form, this should only be used if a user has Javascript disabled
        should_redirect = False
        if request.form != None and len(request.form) > 0:
            data = request.form
            should_redirect = True
        else:
            data = request.json

        if data.get("document") == None:
            return "Invalid request. No document found.", 400

        classification = classify(data["document"])
        print(classification["confidence"])
        if should_redirect == False:
            return jsonify(
                { 
                "document": data["document"],
                "classification": classification["value"],
                "confidence": classification["confidence"]
                }
            )
        else:
            return render_template("result.html", result_text=classification)
    # web app requests
    elif request.method == "GET":
        logging.info("INFO:Get request to Spam Predictor Model")
        return render_template(
            "text-input.html",
            title="Spam Email Detector",
            input_label="Email",
            button_text="Check If Spam",
            data_href="https://www.kaggle.com/datasets/veleon/ham-and-spam-dataset",
            data_link_text="Kaggle.com",
        )
    else:
        return "Invalid request. Only GET and POST requests are supported."

if __name__ == "__main__":
    app.run()