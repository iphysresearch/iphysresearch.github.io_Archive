from flask_frozen import Freezer
from app import app
import sys

freezer = Freezer(app)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        # freezer.freeze()
        freezer.run(debug=True)
    else:
        app.run(port=5000)