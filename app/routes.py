from flask import render_template
from flask_frozen import Freezer
from app import app
from flaskext.markdown import Markdown  # https://pythonhosted.org/Flask-Markdown/

import markdown, datetime, sys, codecs
from app.extensions.footnotes_dt import FootnoteExtension
from app.extensions.dt_cite import CiteExtension
from app.extensions.dt_code import FencedCodeExtension
from app.extensions.meta_data import MetaExtension
from app.extensions.appendix import AppendixExtension
from app.extensions.checkbox import ChecklistExtension

meta_keys = ['title', 'description', 'authors', 'authors_url', 
             'affiliations', 'affiliations_url','publisheddate']

@app.route('/')
def index():
    with codecs.open('post/template.md', "r", "utf-8") as f:
        text = f.read()
        # Markdown(app, extensions = [CiteExtension(), FootnoteExtension(),'meta'], output_format="html5", )
        md = markdown.Markdown(extensions = [MetaExtension(), AppendixExtension()], output_format="html5", )
        html = md.convert(text)
    Markdown(app, extensions = [CiteExtension(), FootnoteExtension(), FencedCodeExtension(), AppendixExtension(), ChecklistExtension(), 'meta','nl2br','tables'], output_format="html5", )
    # print(dir(md))
    # html = md.convertFile('app/post/template.md')
    meta_data = md.Meta
    for key in meta_keys:
        meta_data[key] = md.Meta.setdefault(key, ['No {}'.format(key)])
    meta_data['updatedDate'] = datetime.datetime.today()
    # print(meta_data)
    return render_template('template.html', meta_data=meta_data, text=text, appendix=md.Appendix+'\n---\n---')
    

freezer = Freezer(app)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        freezer.freeze()
    else:
        app.run(port=5000)
# text = '---'.join(text.split('---')[:-2])

# export FLASK_APP=microblog.py    

# md = markdown.Markdown(extensions = ['meta'])
# html = md.convertFile('app/post/template.md')
# md.Meta # https://python-markdown.github.io/extensions/meta_data/

# import markdown
# from extensions.footnotes_dt import FootnoteExtension
# md = markdown.Markdown(extensions = [FootnoteExtension(),'meta'])
# html = md.convertFile('../app/post/template.md')
