import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';
import './MarkdownRenderer.css';


const applyOutsideMath = (text, fn) => {
  const re = /\$\$[\s\S]*?\$\$|\$[^$\n]*?\$/g;
  const parts = [];
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(fn(text.slice(last, m.index)));
    parts.push(m[0]);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(fn(text.slice(last)));
  return parts.join('');
};

const fixCollapsedNewlines = (body) => body.replace(/[ \t]\\+[ \t]+/g, ' \\\\ ');

const looksLikeFormulaLine = (t) => {
  if (!t || t.length > 120) return false;
  if (t.startsWith('$') || t.startsWith('\\begin') || t.startsWith('\\end')) return false;
  if (t.startsWith('```')) return false;

  const hasEq = t.includes('=');
  const hasCmd = /\\[a-zA-Z]+/.test(t);
  const hasSubSup = /[_^]\s*[\{A-Za-z0-9]/.test(t);

  if (!hasEq && !hasCmd) return false;
  if (!hasCmd && !hasSubSup) return false;

  const mathChars = (t.match(/[\\{}_^()\[\]+\-*/=<>|]/g) || []).length;
  if (mathChars / t.length < 0.15) return false;

  return true;
};


const LOOSE_ALIGNED_RE =
  /(^|[\s\(\[\{"'“”>])((?:[A-Za-z0-9_^{}()\[\]+\-*\/.,=<>|]{1,40}?[ \t]*&[ \t]*=[^\\\n$]{1,200}?)(?:[ \t]\\+[ \t]+(?:[A-Za-z0-9_^{}()\[\]+\-*\/.,=<>|]{1,40}?[ \t]*&[ \t]*=[^\\\n$]{1,200}?))+)/g;

const wrapLooseAligned = (segment) =>
  segment.replace(LOOSE_ALIGNED_RE, (full, lead, body) => {
    //  must contain some LaTeX-ish syntax
    if (!/[_^\\]/.test(body)) return full;

    //  must have at least two `& =`
    const pairs = body.match(/&[ \t]*=/g);
    if (!pairs || pairs.length < 2) return full;

    // the LHS of the first equation must not be English prose.
    //     Strip \commands, then reject if any 4+ letter word remains.
    const lhs = body.split(/&[ \t]*=/)[0];
    const lhsNoCmds = lhs.replace(/\\[A-Za-z]+/g, '');
    if (/\b[a-z]{4,}\b/i.test(lhsNoCmds)) return full;

    // split into individual equations on the " \ " separators
    const parts = body
      .split(/[ \t]\\+[ \t]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (parts.length < 2) return full;

    const aligned = parts.join(' \\\\ ');

    return `${lead}\n$$\n\\begin{aligned}\n${aligned}\n\\end{aligned}\n$$\n`;
  });

const normalizeMath = (text) => {
  if (!text) return '';
  let out = text;

  // 1. \[ ... \]  ->  $$ ... $$
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, (_, b) => `\n$$\n${b.trim()}\n$$\n`);

  // 2. \( ... \)  ->  $ ... $
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_, b) => `$${b.trim()}$`);

  // 3. Bare \begin{env}...\end{env} (outside math) -> fix \\ + wrap in $$
  out = applyOutsideMath(out, (seg) =>
    seg.replace(
      /\\begin\{(aligned|align\*?|gather\*?|multline\*?|matrix|bmatrix|pmatrix|vmatrix|Vmatrix|smallmatrix|cases|array)\}([\s\S]*?)\\end\{\1\}/g,
      (_, env, body) =>
        `\n$$\n\\begin{${env}}${fixCollapsedNewlines(body)}\\end{${env}}\n$$\n`
    )
  );

  out = applyOutsideMath(out, wrapLooseAligned);

  out = applyOutsideMath(out, (seg) =>
    seg
      .split('\n')
      .map((line) => {
        const t = line.trim();
        if (!looksLikeFormulaLine(t)) return line;
        return `$$\n${t}\n$$`;
      })
      .join('\n')
  );

  return out;
};


export const MarkdownRenderer = ({ content }) => {
  if (!content) return null;
  const normalized = normalizeMath(content);

  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[
          [rehypeKatex, { throwOnError: false, strict: false, output: 'html' }],
          [rehypeHighlight, { detect: true, ignoreMissing: true }],
        ]}
        components={{
          a: ({ node, ...props }) => (
            <a target="_blank" rel="noopener noreferrer" {...props} />
          ),
          code: ({ node, className, children, ...props }) => {
            const isBlock =
              /language-/.test(className || '') ||
              String(children).includes('\n');
            if (!isBlock) {
              return (
                <code className="inline-code" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          table: ({ node, ...props }) => (
            <div className="table-wrapper">
              <table {...props} />
            </div>
          ),
        }}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  );
};