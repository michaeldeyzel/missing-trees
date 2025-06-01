from flask import Flask, request, jsonify
from get_missing_trees import missing_trees_orchard

app = Flask(__name__)

@app.route('/orchards/<int:n>/missing-trees', methods=['GET'])
def data(n):
    try:
        # n = int(request.args.get('n'))
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header, Please provide 'Bearer <your-token>' field for Authorization key"}), 401

        result = missing_trees_orchard(n, auth_header)
        return jsonify(result)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing parameter 'n'"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Important: listen on all interfaces
