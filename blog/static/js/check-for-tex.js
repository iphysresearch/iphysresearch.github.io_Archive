(function () {
  var body = document.body.textContent;
  if (body.match(/(?:\$|\\\(|\\\[|\\begin\{.*?})/)) {
    if (!window.MathJax) {
      window.MathJax = {
        tex: {
          // http://docs.mathjax.org/en/latest/options/input/tex.html
          packages: ['base'],        // extensions to use
          inlineMath: {'[+]': [['$', '$']]}, // start/end delimiter pairs for in-line math
          displayMath: [             // start/end delimiter pairs for display math
            ['$$', '$$'],
            ['\\[', '\\]']
          ],
          tags: 'all',               // 'none' or 'ams' or 'all'
          processEscapes: true,      // use \$ to produce a literal dollar sign
          processEnvironments: true, // process \begin{xxx}...\end{xxx} outside math mode
          processRefs: true,         // process \ref{...} outside of math mode
          digits: /^(?:[0-9]+(?:\{,\}[0-9]{3})*(?:\.[0-9]*)?|\.[0-9]+)/,
          tagSide: 'right',          // side for \tag macros
          tagIndent: '0.8em',        // amount to indent tags      
          useLabelIds: true,         // use label name rather than tag for ids
        }
      };
    }
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
    document.head.appendChild(script);
  }
})();