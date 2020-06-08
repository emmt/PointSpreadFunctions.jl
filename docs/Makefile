SOURCES = \
    make.jl \
    src/index.md \
    src/library.md

default: build/index.html

clean:
	rm -f *~ */*~ */*/*~

force:
	julia --color=yes make.jl

build/index.html: $(SOURCES)
	julia --color=yes make.jl

.PHOONY: default clean force
