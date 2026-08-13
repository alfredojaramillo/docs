# Notes on the Likelihood Function

LaTeX notes developing the likelihood function from the binomial example.

## Layout

```
.
├── probability-and-statistics.tex    top-level document
├── preamble.tex                      packages + notation macros
├── sections/
│   ├── 01-binomial-shape.tex         shape of L(q | v, n); log-concavity; MLE q = v/n
│   ├── 02-same-functional-form.tex   PMF and likelihood as orthogonal slices of one surface
│   └── 03-definition.tex             measure-theoretic definition (Radon–Nikodym) + iid case
├── Makefile
└── .gitignore
```

## Notation

Conventions are fixed in `preamble.tex`. Notably, Lebesgue measure is `\Leb`
(rendered as $\mu$, never $\lambda$). Redefine the macro there to change it
everywhere.

## Build

```
make          # compile to probability-and-statistics.pdf via latexmk
make clean    # remove build artifacts, keep the PDF
make cleanall # remove build artifacts and the PDF
```

Requires a TeX distribution with `latexmk` and `pdflatex`.
