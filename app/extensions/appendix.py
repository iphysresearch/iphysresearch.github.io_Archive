"""
Meta Data Extension for Python-Markdown
=======================================
This extension adds Meta Data handling to markdown.
See <https://Python-Markdown.github.io/extensions/meta_data>
for documentation.
Original code Copyright 2007-2008 [Waylan Limberg](http://achinghead.com).
All changes Copyright 2008-2014 The Python Markdown Project
License: [BSD](https://opensource.org/licenses/bsd-license.php)
"""

from __future__ import absolute_import
from __future__ import unicode_literals
# from . import Extension
from markdown.extensions import Extension
# from ..preprocessors import Preprocessor
from markdown.preprocessors import Preprocessor
import re
import logging

log = logging.getLogger('MARKDOWN')

# Global Vars
META_RE = re.compile(r'^[ ]{0,3}(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)')
META_MORE_RE = re.compile(r'^[ ]{4,}(?P<value>.*)')
BEGIN_RE = re.compile(r'^-{3}(\s.*)?')
END_RE = re.compile(r'^(-{3}|\.{3})(\s.*)?')

# (\s.*)?-{3}$
class AppendixExtension (Extension):
    """ Appendix-Data extension for Python-Markdown. """

    def extendMarkdown(self, md):
        """ Add AppendixPreprocessor to Markdown instance. """
        md.registerExtension(self)
        self.md = md
        md.preprocessors.register(AppendixPreprocessor(md), 'appendix', 27)

    def reset(self):
        self.md.Appendix = {}


class AppendixPreprocessor(Preprocessor):
    """ Get Appendix-Data. """

    def run(self, lines):
        """ Parse Appendix-Data and store in Markdown.Appendix. """
        lines.reverse()
        appendix = []
        # print('lines_begin:', lines)
        while True:
            if lines[0]:
                break
            else:
                lines.pop(0)
        # print('lines_beforeif:', lines)
        if lines and BEGIN_RE.match(lines[0]):
            lines.pop(0)
        # print('lines_beforewhile:', lines)
        while lines:
            line = lines.pop(0)
            # m1 = META_RE.match(line)
            if END_RE.match(line):
                break  # blank line or end of YAML header - done
            # print(line)
            appendix.append(line)
            # if m1:
            #     key = m1.group('key').lower().strip()
            #     value = m1.group('value').strip()
            #     try:
            #         appendix[key].append(value)
            #     except KeyError:
            #         appendix[key] = [value]
            # else:
            #     m2 = META_MORE_RE.match(line)
            #     if m2 and key:
            #         # Add another line to existing key
            #         appendix[key].append(m2.group('value').strip())
            #     else:
            #         lines.insert(0, line)
            #         break  # no appendix data - done
        appendix.reverse()
        self.md.Appendix = '\n'.join(appendix)
        lines.reverse()
        # print('appendex_py:', appendix)
        # print('lines_end:', lines)
        return lines


def makeExtension(**kwargs):  # pragma: no cover
    return AppendixExtension(**kwargs)