from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index_get():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    try:
        user_text = request.get_json().get("msg")
        if not user_text:
            return jsonify({"error": "No message provided"}), 400
        response = get_response(user_text)
        message = {"answer": response}
        return jsonify(message)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
