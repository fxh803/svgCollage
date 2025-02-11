from flask import Flask, request

app = Flask(__name__)

@app.route('/receive', methods=['POST'])
def receive_message():
    if request.method == 'POST':
        data = request.get_json()  # 获取 POST 请求的 JSON 数据
        return data  # 返回接收到的 JSON 数据

if __name__ == '__main__':
    app.run(debug=True)