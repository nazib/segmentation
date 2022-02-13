import os
import flask
from flask import jsonify
from train import trainer
from predict import predict
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
    #data_dir,epochs,size,batch_size,lr,val_percent
    status = trainer(data["dir"],data["epochs"],data["dimension"],data["batch_size"],data['lr'],data["validation"])
    return jsonify(status)

@app.route('/start_predict', methods=['POST'])
def model_predict():
    data = flask.request.get_json(force=True)
    #data_dir,epochs,size,batch_size,lr,val_percent
    status = predict(data["data_dir"],data["image_dim"],data["model_dir"],data["with_label"])
    return jsonify(status)

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    #http_server = WSGIServer(('0.0.0.0', port), create_app('production'))
    #http_server.serve_forever()
    #app.logger.info("App started ")
    app.run(host='0.0.0.0', port=port)