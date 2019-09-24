from __future__ import absolute_import
from __future__ import unicode_literals
import markdown
from markdown import Extension
from markdown.preprocessors import Preprocessor
from markdown.inlinepatterns import Pattern


CUSTOM_CLS_RE = r'[!]{2}(?P<class>.+)[|](?P<text>.+)[!]{2}'


class CiteExtension(Extension):
    """ Extension class for markdown """

    def extendMarkdown(self, md, md_globals):
        md.inlinePatterns["custom_span_class"] = CustomSpanClassPattern(CUSTOM_CLS_RE, md)

class CustomSpanClassPattern(Pattern):

    def handleMatch(self, matched):

        """
        If string matched
        regexp expression create
        new span elem with given class
        """

        cls = matched.group("class")
        text = matched.group("text")

        elem = markdown.util.etree.Element("d-cite")
        elem.set("key", cls)
        elem.text = markdown.util.AtomicString(text)
        return elem

def makeExtension(*args, **kwargs):
    return CustomSpanClassExtension(*args, **kwargs)