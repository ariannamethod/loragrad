# loragrad — phase 1 (mechanism) + phase 2 (training with routing)
#
# Cross-platform: macOS uses Accelerate framework, Linux uses OpenBLAS.
# notorch.c + notorch.h are vendored so the project builds self-contained.

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
CFLAGS  += -I.

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK
  BLAS_LIBS  = -framework Accelerate
endif

ifeq ($(UNAME), Linux)
  ifneq ($(shell command -v pkg-config 2>/dev/null),)
    BLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
    BLAS_LIBS   := $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)
  else
    BLAS_CFLAGS :=
    BLAS_LIBS   := -lopenblas
  endif
  BLAS_FLAGS = -DUSE_BLAS $(BLAS_CFLAGS)
  # rdynamic + -g so backtrace_symbols_fd produces readable frames in
  # the Railway logs when something segfaults.
  CFLAGS += -rdynamic -g -D_GNU_SOURCE
endif

LDLIBS ?= -lm

.PHONY: all clean smoke train

all: loragrad.o loragrad_standalone.o examples/smoke_test examples/train_loragrad

# Vendored notorch — built once, linked into the trainer.
notorch.o: notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -c -o $@ notorch.c

# Standalone object — no notorch include. Provides text/buffer sketchers
# plus full field/voting machinery; lg_signature_from_grads is a stub.
loragrad_standalone.o: loragrad.c loragrad.h
	$(CC) $(CFLAGS) -DLG_STANDALONE -c -o $@ loragrad.c

# Notorch-linked object — full version.
loragrad.o: loragrad.c loragrad.h notorch.h
	$(CC) $(CFLAGS) -c -o $@ loragrad.c

examples/smoke_test: examples/smoke_test.c loragrad_standalone.o loragrad.h
	$(CC) $(CFLAGS) -o $@ examples/smoke_test.c loragrad_standalone.o $(LDLIBS)

examples/train_loragrad: examples/train_loragrad.c loragrad.o notorch.o loragrad.h notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o $@ examples/train_loragrad.c loragrad.o notorch.o $(BLAS_LIBS) $(LDLIBS)

smoke: examples/smoke_test
	./examples/smoke_test

train: examples/train_loragrad
	./examples/train_loragrad --routed

clean:
	rm -f loragrad.o loragrad_standalone.o notorch.o examples/smoke_test examples/train_loragrad
