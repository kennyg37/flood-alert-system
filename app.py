import os
from flask import Flask
import numpy as np
from dotenv import load_dotenv

load_dotenv()

port = int(os.environ.get('PORT'))

app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug=True, port=port)
