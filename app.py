import os
import flask
from flask import jsonify

app = flask.Flask(__name__)

@app.errorhandler(400)
def value_error(e):
    return jsonify(error=str(e)), 400

@app.route('/health', methods=['GET'])
def health_check():
    status = {200: "Container running successfully"}
    return jsonify(status)

@app.route('/start_training', methods=['POST'])
def model_train():
    data = flask.request.get_json(force=True)
    data={100:"Testing"}
    return jsonify(data)


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    #http_server = WSGIServer(('0.0.0.0', port), create_app('production'))
    #http_server.serve_forever()
    app.logger.info("App started ")
    app.run(host='0.0.0.0', port=port)