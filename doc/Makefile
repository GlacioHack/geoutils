# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	echo "Removing build files..."
	if [ -d "$(BUILDDIR)" ]; then rm -r "$(BUILDDIR)"; fi
	if [ -d "$(SOURCEDIR)/auto_examples" ]; then rm -r "$(SOURCEDIR)/auto_examples"; fi
	if [ -d "$(SOURCEDIR)/gen_modules" ]; then rm -r "$(SOURCEDIR)/gen_modules"; fi

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
