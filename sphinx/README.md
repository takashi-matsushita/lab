# sphinx

#### test of sphinx/rst

__can generate HTML and PDF files with the following commands__

```shell
make html
pandoc -V CJKmainfont=IPAexGothic --pdf-engine=xelatex source/rst-cheetsheet.rst -o test.pdf
# rst2pdf --stylesheets cheatsheet-jp.json source/rst-cheetsheet.rst # rst2pdf doesn't work with python3
```
