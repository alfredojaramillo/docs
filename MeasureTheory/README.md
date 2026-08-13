# Notes on Measure Theory

LaTeX notes on absolute continuity and the Radon–Nikodym theorem, with the
Lebesgue decomposition and two instances (probability densities and conditional
expectation).

## Layout

```
.
├── main.tex                              top-level document
├── preamble.tex                          packages + notation macros
├── sections/
│   ├── 01-absolute-continuity.tex        ≪ and ⊥; ε–δ form; Jordan decomposition
│   ├── 02-radon-nikodym.tex              the theorem + calculus of dν/dμ
│   └── 03-lebesgue-decomposition.tex     ν = ν_ac + ν_s; densities; E[X | G]
├── Makefile
└── .gitignore
```

## Notation

Conventions are fixed in `preamble.tex`: `\abscont` for $\ll$, `\sing` for
$\perp$, and `\RN{\nu}{\mu}` for the Radon–Nikodym derivative $d\nu/d\mu$. These
match the companion likelihood notes in `../ProbabilityAndStatistics`, where the
density $f(x \mid \theta)$ is exactly such a derivative.

## Build

```
make          # compile to main.pdf via latexmk
make clean    # remove build artifacts, keep the PDF
make cleanall # remove build artifacts and the PDF
```

Requires a TeX distribution with `latexmk` and `pdflatex`.
