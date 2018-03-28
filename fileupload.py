from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
from sample import local

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)



@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/load', methods = ['GET', 'POST'])
def load_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save('imageconv/myimage.jpg')
      os.system("./example.sh")
      w = local()
      print w
      f.save(secure_filename(f.filename))
      return render_template('upload.html',output = w)



if __name__ == '__main__':
   app.run(debug = True)
